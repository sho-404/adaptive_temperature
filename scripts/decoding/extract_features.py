"""Extract per-prompt features for the adaptive-temperature MLP (ONE language).

PER-SEQUENCE setup: exactly one feature vector per prompt, taken at the FINAL
prompt token (the decision point, before any answer token is generated). The
same extractor is used for BOTH train and eval (--split), so the feature
definition is byte-for-byte identical on both ends — that is the whole point of
having one script. The HF forward pass (eager attention) is the ONLY way to get
the hidden state + attention weights; vLLM cannot produce them.

Features stored per prompt (each kept separately so ablations can compose subsets):
  S1  entropy        - Shannon entropy of the next-token softmax        (scalar)
  S2  logit_gap      - logit[rank 1] - logit[rank k]   (k=5)            (scalar)
  S3  hidden         - last-layer hidden at final prompt token, full 3072, NO projection
  S4  attn_entropy   - mean over last-layer heads of the query's attention entropy (RAW)
  (seq_len is also stored so S4 can optionally be normalised by log(seq_len) at train time)

NOTE: S5 (relative position) is intentionally absent — it is undefined per-sequence
(no generated tokens exist at decision time).

Output: features/gsm8k_<lang>.pt   (gitignored; lives on the run machine)

Usage:
    python extract_features.py --lang en --validate-one   # PRE-FLIGHT: 1 example, hard asserts, no save
    python extract_features.py --lang en                  # full run over the pair questions
    python extract_features.py --lang en --split test     # eval-time features (test split)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))  # scripts/: model.py, tasks/
from model import load_config, get_device, get_dtype, load_model, encode
import tasks.gsm8k as gsm8k

DATA = Path(__file__).parents[2] / "data"
LOGIT_K = 5  # S2: gap between the 1st- and k-th-ranked logit


# ============================================================
# Prompt source
# ============================================================

def load_prompts(lang: str, split: str) -> list[dict]:
    """Return [{index, question}, ...] deduplicated by index, in index order.

    train -> the exact questions that produced preference pairs (datasets/gsm8k_<lang>.jsonl),
             so we featurise precisely the prompts we will train on.
    test  -> the held-out split via the task loader (same path eval_fixed uses).
    """
    if split == "train":
        path = DATA / f"gsm8k_{lang}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Pairs file not found: {path}")
        by_index: dict[int, str] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_index[int(rec["index"])] = rec["question"]
        return [{"index": i, "question": by_index[i]} for i in sorted(by_index)]

    examples = gsm8k.load(lang, split)
    return [{"index": int(ex["index"]), "question": ex["question"]} for ex in examples]


def template_hash() -> str:
    """Stable hash of the prompt template, stamped into the cache so eval can assert
    its features were built with the IDENTICAL formatting (the consistency guardrail)."""
    probe = gsm8k.build_messages("__TEMPLATE_PROBE__")
    blob = json.dumps(probe, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ============================================================
# Feature extraction (the single source of truth)
# ============================================================

@torch.no_grad()
def extract(model, tokenizer, question: str, device: torch.device, cfg: dict) -> dict:
    """One HF forward pass over the prompt -> the 4 signals at the final prompt token.

    Raises if attentions are missing (S4 would otherwise be a silent dead feature).
    """
    inputs = encode(tokenizer, gsm8k.build_messages(question), device, cfg["max_prompt_tokens"])
    out = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )

    if getattr(out, "attentions", None) is None:
        raise RuntimeError(
            "Model returned no attentions. S4 (attention entropy) cannot be computed. "
            "Ensure attn_implementation='eager' (config use_attention_entropy=true). "
            "Refusing to silently emit zeros."
        )

    # S3: last layer, final prompt token, full hidden (no projection).
    hidden = out.hidden_states[-1][:, -1, :].float().squeeze(0).cpu()

    # S1 / S2: next-token distribution at the final position.
    logits = out.logits[:, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).item()
    topk = torch.topk(logits, k=LOGIT_K, dim=-1).values.squeeze(0)
    logit_gap = (topk[0] - topk[LOGIT_K - 1]).item()  # rank1 - rank5

    # S4: last-layer attention of the final-token query, per-head entropy, averaged (RAW).
    attn = out.attentions[-1][0, :, -1, :].float().clamp_min(1e-12)  # [heads, keys]
    attn_entropy = (-(attn * attn.log()).sum(dim=-1)).mean().item()

    attention_mask = inputs.get("attention_mask")
    seq_len = int(attention_mask.sum().item()) if attention_mask is not None else int(inputs["input_ids"].shape[-1])

    return {
        "hidden": hidden,            # [3072]
        "entropy": entropy,          # S1
        "logit_gap": logit_gap,      # S2 (k=5)
        "attn_entropy": attn_entropy,  # S4 (raw)
        "seq_len": seq_len,
    }


# ============================================================
# Pre-flight: validate ONE example
# ============================================================

def validate_one(model, tokenizer, prompts: list[dict], device: torch.device, cfg: dict) -> None:
    ex = prompts[0]
    print(f"\n=== PRE-FLIGHT: validating one example (index={ex['index']}) ===")
    feat = extract(model, tokenizer, ex["question"], device, cfg)

    hidden = feat["hidden"]
    expected = int(cfg["hidden_size"])
    ok_dim = hidden.shape[0] == expected
    ok_finite = all(
        torch.isfinite(torch.tensor(float(feat[k]))).item()
        for k in ("entropy", "logit_gap", "attn_entropy")
    ) and bool(torch.isfinite(hidden).all().item())

    print(f"  S3 hidden  : shape={tuple(hidden.shape)}  expected=({expected},)  -> {'OK' if ok_dim else 'MISMATCH'}")
    print(f"  S1 entropy : {feat['entropy']:.4f}")
    print(f"  S2 logitgap: {feat['logit_gap']:.4f}  (k={LOGIT_K}: rank1 - rank{LOGIT_K})")
    print(f"  S4 attn_ent: {feat['attn_entropy']:.4f}  (raw, mean over heads)")
    print(f"  seq_len    : {feat['seq_len']}")
    print(f"  all finite : {'OK' if ok_finite else 'NON-FINITE VALUES'}")

    assert ok_dim, f"hidden dim {hidden.shape[0]} != config hidden_size {expected}"
    assert ok_finite, "non-finite feature value(s) detected"
    assert feat["attn_entropy"] > 0.0, "attn_entropy is 0 — attention likely not produced; S4 would be dead"
    print("\n  ALL ASSERTS PASSED — features look sane. Safe to run the full extraction.\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None, help="process only the first N prompts")
    ap.add_argument("--validate-one", action="store_true", help="pre-flight: 1 example, assert, no save")
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.get("use_attention_entropy"):
        raise SystemExit("config use_attention_entropy must be true (S4 needs eager attention).")

    device = get_device(cfg.get("device", "auto"))
    dtype = get_dtype(cfg["dtype_name"])

    print(f"Lang / split : {args.lang} / {args.split}")
    print(f"Device/dtype : {device} / {dtype}")

    tokenizer, model = load_model(cfg, device, dtype)
    prompts = load_prompts(args.lang, args.split)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    print(f"Prompts      : {len(prompts)}")

    if args.validate_one:
        validate_one(model, tokenizer, prompts, device, cfg)
        return

    n = len(prompts)
    indices = torch.empty(n, dtype=torch.long)
    hidden = torch.empty(n, int(cfg["hidden_size"]), dtype=torch.float32)
    entropy = torch.empty(n, dtype=torch.float32)
    logit_gap = torch.empty(n, dtype=torch.float32)
    attn_entropy = torch.empty(n, dtype=torch.float32)
    seq_len = torch.empty(n, dtype=torch.long)

    start = time.perf_counter()
    for i, ex in enumerate(prompts):
        feat = extract(model, tokenizer, ex["question"], device, cfg)
        indices[i] = ex["index"]
        hidden[i] = feat["hidden"]
        entropy[i] = feat["entropy"]
        logit_gap[i] = feat["logit_gap"]
        attn_entropy[i] = feat["attn_entropy"]
        seq_len[i] = feat["seq_len"]

        if (i + 1) % 100 == 0 or (i + 1) == n:
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed
            eta = (n - (i + 1)) / rate if rate > 0 else 0.0
            print(f"  {i + 1}/{n}  ({rate:.1f}/s, eta {eta / 60:.1f}m)")

    payload = {
        "index": indices,
        "hidden": hidden,
        "entropy": entropy,
        "logit_gap": logit_gap,
        "attn_entropy": attn_entropy,
        "seq_len": seq_len,
        "meta": {
            "lang": args.lang,
            "split": args.split,
            "n": n,
            "hidden_size": int(cfg["hidden_size"]),
            "logit_k": LOGIT_K,
            "signals": ["S1_entropy", "S2_logit_gap_k5", "S3_hidden_full", "S4_attn_entropy_raw"],
            "model": cfg["model_name_or_path"],
            "dtype": cfg["dtype_name"],
            "device": str(device),
            "attn_impl": "eager",
            "template_hash": template_hash(),
            "max_prompt_tokens": int(cfg["max_prompt_tokens"]),
        },
    }

    out_dir = DATA
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gsm8k_{args.lang}.pt"
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)

    elapsed = time.perf_counter() - start
    print(f"\nDone. {n} prompts in {elapsed / 60:.1f}m.")
    print(f"Saved -> {out_path}")
    print(f"Template hash: {payload['meta']['template_hash']}")


if __name__ == "__main__":
    main()
