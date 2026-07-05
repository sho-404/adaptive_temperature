"""ORACLE PROBE: extract features OVER THE REASONING (not just the prompt).

Motivation. The per-sequence pipeline featurises only the final PROMPT token,
before any reasoning. On GSM8K that signal is weak: easy-vs-hard is barely
separable from the prompt (5-fold AUC ~0.59), so the scorer can't beat the
trivial "always greedy" bar (86.7%). Hypothesis: difficulty for math becomes
visible DURING the reasoning, not at the prompt. A token-level decoder would
see it (it conditions on h_t, which encodes the reasoning so far).

This script tests that hypothesis cheaply, WITHOUT regeneration: every pair
already stores a chosen (usually correct) and rejected (usually incorrect)
response. We teacher-force each stored response through the frozen model and
grab the last-layer hidden state over the RESPONSE region:
  hidden_last  - hidden at the final response token (end of reasoning)
  hidden_mean  - mean-pooled hidden over the response tokens

Crucially, for a single prompt the chosen and rejected responses share the
SAME prompt features, so prompt-only features can't separate them by
construction. If reasoning features can predict `was_correct` with high AUC,
the signal is there and token-level is worth building. If not, we save the work.

Label per row: was_correct (whether that response reached the right answer).

Output: features/gsm8k_<lang>_reasoning.pt   (gitignored; lives on the run box)

Usage:
    python extract_reasoning_features.py --lang en --validate-one   # 1 chosen + 1 rejected, assert, no save
    python extract_reasoning_features.py --lang en --limit 50       # small smoke run
    python extract_reasoning_features.py --lang en                  # full run (chosen + rejected per pair)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model import load_config, get_device, get_dtype, load_model
import tasks.gsm8k as gsm8k
from extract_features import template_hash

HERE = Path(__file__).parent
FRACS = [0.25, 0.5, 0.75, 1.0]  # positional taps along the reasoning (1.0 = end)


# ============================================================
# Data
# ============================================================

def load_pairs(lang: str) -> list[dict]:
    path = HERE / "datasets" / f"gsm8k_{lang}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_rows(pairs: list[dict]):
    """Yield (index, side, mode, temperature, was_correct, question, response).

    side: 0 = chosen, 1 = rejected. Both sides of every pair are emitted
    (all_fail included; filter on `mode` at analysis time).
    """
    for p in pairs:
        for side, key in ((0, "chosen"), (1, "rejected")):
            r = p[key]
            yield (int(p["index"]), side, p["mode"],
                   float(r["temperature"]), bool(r["was_correct"]),
                   p["question"], r["response"])


# ============================================================
# Feature extraction (teacher-forced over prompt + response)
# ============================================================

def _ids(tokenizer, messages, **kw) -> torch.Tensor:
    out = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, **kw)
    return out["input_ids"]


def _common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    """Length of the shared leading token run (robust way to locate the response
    region without assuming the chat template's generation-prompt tokens)."""
    m = min(a.shape[-1], b.shape[-1])
    eq = (a[0, :m] == b[0, :m])
    nz = (~eq).nonzero()
    return int(nz[0].item()) if len(nz) else m


@torch.no_grad()
def extract(model, tokenizer, question: str, response: str, device, cfg: dict) -> dict:
    """One forward pass over prompt+response -> hidden over the response region."""
    prompt_msgs = gsm8k.build_messages(question)
    full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]

    prompt_ids = _ids(tokenizer, prompt_msgs)
    # continue_final_message=True: append the assistant response as a prefix
    # being continued -> no trailing end-of-turn token, last token = end of reasoning.
    full_ids = _ids(tokenizer, full_msgs, continue_final_message=True)

    prompt_len = _common_prefix_len(prompt_ids, full_ids)
    # Cap the total length defensively (prompt budget + generation budget).
    max_total = int(cfg["max_prompt_tokens"]) + int(cfg["max_new_tokens"])
    if full_ids.shape[-1] > max_total:
        full_ids = full_ids[:, :max_total]

    full_ids = full_ids.to(device)
    out = model(input_ids=full_ids, use_cache=False, output_hidden_states=True, return_dict=True)

    hs = out.hidden_states[-1][0].float()        # [seq, hidden]
    resp = hs[prompt_len:, :]                     # response region
    if resp.shape[0] == 0:                        # degenerate: empty response region
        resp = hs[-1:, :]

    # Positional taps: hidden at 25/50/75/100% through the reasoning region.
    r = resp.shape[0]
    frac_rows = [resp[min(r - 1, max(0, int(round(f * r)) - 1))] for f in FRACS]

    return {
        "hidden_frac": torch.stack(frac_rows).cpu(),  # [len(FRACS), hidden]
        "hidden_last": resp[-1].cpu(),            # end of reasoning (== frac 1.0)
        "hidden_mean": resp.mean(dim=0).cpu(),    # pooled over reasoning
        "resp_len": int(resp.shape[0]),
        "prompt_len": int(prompt_len),
        "full_len": int(full_ids.shape[-1]),
        "prefix_is_clean": bool(prompt_len == prompt_ids.shape[-1]),
    }


# ============================================================
# Pre-flight
# ============================================================

def validate_one(model, tokenizer, pairs, device, cfg) -> None:
    # Exercise both a correct (chosen) and an incorrect (rejected) response.
    rows = list(iter_rows(pairs))
    chosen = next(r for r in rows if r[1] == 0 and r[4])          # side=chosen, correct
    rejected = next(r for r in rows if r[1] == 1 and not r[4])    # side=rejected, incorrect
    expected = int(cfg["hidden_size"])

    for tag, row in (("CHOSEN/correct", chosen), ("REJECTED/incorrect", rejected)):
        idx, side, mode, temp, correct, q, resp = row
        print(f"\n=== PRE-FLIGHT {tag}  (index={idx}, mode={mode}, tau={temp}, was_correct={correct}) ===")
        feat = extract(model, tokenizer, q, resp, device, cfg)
        hl, hm = feat["hidden_last"], feat["hidden_mean"]
        ok_dim = hl.shape[0] == expected and hm.shape[0] == expected
        ok_finite = bool(torch.isfinite(hl).all() and torch.isfinite(hm).all())
        print(f"  prompt_len / full_len : {feat['prompt_len']} / {feat['full_len']}  "
              f"(clean prefix: {feat['prefix_is_clean']})")
        print(f"  response tokens       : {feat['resp_len']}")
        print(f"  hidden_last shape     : {tuple(hl.shape)}  expected=({expected},) -> {'OK' if ok_dim else 'MISMATCH'}")
        print(f"  hidden_mean shape     : {tuple(hm.shape)}")
        print(f"  all finite            : {'OK' if ok_finite else 'NON-FINITE'}")
        assert ok_dim, "hidden dim mismatch"
        assert ok_finite, "non-finite hidden"
        assert feat["resp_len"] > 0, "empty response region — prefix detection failed"

    print("\n  ALL ASSERTS PASSED — reasoning extraction is sane. Safe to run the full job.\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    ap.add_argument("--validate-one", action="store_true", help="pre-flight: 1 chosen + 1 rejected, assert, no save")
    args = ap.parse_args()

    cfg = load_config()
    device = get_device(cfg.get("device", "auto"))
    dtype = get_dtype(cfg["dtype_name"])

    print(f"Lang         : {args.lang}")
    print(f"Device/dtype : {device} / {dtype}")

    tokenizer, model = load_model(cfg, device, dtype)
    pairs = load_pairs(args.lang)
    print(f"Pairs        : {len(pairs)}  (rows = {2*len(pairs)} chosen+rejected)")

    if args.validate_one:
        validate_one(model, tokenizer, pairs, device, cfg)
        return

    rows = list(iter_rows(pairs))
    if args.limit is not None:
        rows = rows[: args.limit]
    n = len(rows)

    H = int(cfg["hidden_size"])
    index = torch.empty(n, dtype=torch.long)
    side = torch.empty(n, dtype=torch.long)
    temperature = torch.empty(n, dtype=torch.float32)
    was_correct = torch.empty(n, dtype=torch.bool)
    resp_len = torch.empty(n, dtype=torch.long)
    hidden_frac = torch.empty(n, len(FRACS), H, dtype=torch.float32)
    hidden_last = torch.empty(n, H, dtype=torch.float32)
    hidden_mean = torch.empty(n, H, dtype=torch.float32)
    modes: list[str] = []

    start = time.perf_counter()
    for i, (idx, sd, mode, temp, correct, q, resp) in enumerate(rows):
        feat = extract(model, tokenizer, q, resp, device, cfg)
        index[i] = idx
        side[i] = sd
        temperature[i] = temp
        was_correct[i] = correct
        resp_len[i] = feat["resp_len"]
        hidden_frac[i] = feat["hidden_frac"]
        hidden_last[i] = feat["hidden_last"]
        hidden_mean[i] = feat["hidden_mean"]
        modes.append(mode)
        if (i + 1) % 100 == 0 or (i + 1) == n:
            el = time.perf_counter() - start
            rate = (i + 1) / el
            print(f"  {i+1}/{n}  ({rate:.1f}/s, eta {(n-(i+1))/rate/60:.1f}m)")

    payload = {
        "index": index, "side": side, "mode": modes, "temperature": temperature,
        "was_correct": was_correct, "resp_len": resp_len,
        "hidden_frac": hidden_frac, "fracs": FRACS,
        "hidden_last": hidden_last, "hidden_mean": hidden_mean,
        "meta": {
            "lang": args.lang, "n": n, "hidden_size": H, "fracs": FRACS,
            "model": cfg["model_name_or_path"], "dtype": cfg["dtype_name"],
            "device": str(device), "template_hash": template_hash(),
            "note": "teacher-forced over stored chosen/rejected responses; label=was_correct",
        },
    }
    out_dir = HERE / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gsm8k_{args.lang}_reasoning.pt"
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)

    el = time.perf_counter() - start
    print(f"\nDone. {n} rows in {el/60:.1f}m.  Saved -> {out_path}")


if __name__ == "__main__":
    main()
