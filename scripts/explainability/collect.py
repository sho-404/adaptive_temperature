"""CROSS-MODEL clean data collection: generations + rich hidden-state taps.

One script, any registry model (--model ministral|qwen|llama), any GSM8K split
(--split train|test). Produces the per-model data the analysis paper needs:

  --stage generate  [VM, vLLM]   sample solutions at TEMPS x K, score against
                    ground truth -> data/gsm8k_en_clean_gens.{model}.{split}.jsonl
  --stage extract   [VM, HF]     teacher-force each stored generation, tap:
                      - last-layer hidden at FRACS of the response (incl. 90/95%)
                      - last-layer hidden at the PRE-ANSWER token (the last token
                        before the final '#### N' line -> "does it know before
                        it commits?")
                      - optional (--layers): a depth sweep, N_LAYER_TAPS evenly
                        spaced layers at mid / end / pre-answer positions
                      - per-token logprobs of the response under the model
                        (mean/min/sum + per-quarter means) -> the confidence
                        baselines, untempered and identical across models
                    -> data/gsm8k_en_clean.{model}.{split}.pt

Conventions:
  - TEMPS and the seed formula match validate_clean.py exactly, so the
    ministral/test run reproduces the earlier clean generations byte-for-byte.
  - train split = fit data, test split = eval data. Run --layers on the test
    split only (it dominates file size and the heatmap is fitted by CV there).
  - hidden states are stored fp16 (analysis upcasts); scalars fp32.

Pre-flight on a fresh model (cheap, catches template/parsing issues):
    python collect.py --model qwen --stage generate --split test --limit 100
    python collect.py --model qwen --stage extract  --split test --validate-one
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))  # scripts/: model.py, tasks/
sys.path.insert(0, str(Path(__file__).parents[1] / "decoding"))  # template_hash
from model import (apply_template, get_device, get_dtype, load_config,
                   load_model, load_vllm, model_dims)
import tasks.gsm8k as gsm8k
from extract_features import template_hash

DATA = Path(__file__).parents[2] / "data"

TEMPS = [(0.0, 1), (0.6, 2), (1.0, 2)]  # (temperature, samples per prompt)
FRACS = [0.25, 0.5, 0.75, 0.9, 0.95, 1.0]  # positional taps (1.0 = end of reasoning)
N_LAYER_TAPS = 8                # evenly spaced layers for the depth sweep
LAYER_POSITIONS = ["mid", "end", "pre"]  # 0.5 / 1.0 / pre-answer


def gens_path(model_key: str, split: str) -> Path:
    return DATA / f"gsm8k_en_clean_gens.{model_key}.{split}.jsonl"


def pt_path(model_key: str, split: str) -> Path:
    return DATA / f"gsm8k_en_clean.{model_key}.{split}.pt"


# ============================================================
# Stage 1 — generate (vLLM, one flat batch over all requests)
# ============================================================

def stage_generate(model_key: str, split: str, limit: int | None) -> None:
    from vllm import SamplingParams

    cfg = load_config(model_key)
    examples = gsm8k.load("en", split)
    if limit:
        examples = examples[:limit]
    print(f"Model: {model_key} ({cfg['model_name_or_path']})")
    print(f"Split: {split}  prompts: {len(examples)}  temps: {TEMPS}")

    conversations, meta = [], []
    for ex in examples:
        msgs = gsm8k.build_messages(ex["question"])
        for ti, (t, k) in enumerate(TEMPS):
            for s in range(k):
                # Same formula as validate_clean.py -> ministral/test reproduces.
                seed = int(cfg["seed"]) + ex["index"] * 100 + ti * 10 + s
                conversations.append(msgs)
                meta.append((ex, t, seed))

    sampling = [
        SamplingParams(temperature=max(float(t), 0.0), top_p=1.0,
                       max_tokens=int(cfg["max_new_tokens"]), seed=seed, n=1)
        for (_, t, seed) in meta
    ]

    llm = load_vllm(cfg)
    start = time.perf_counter()
    outputs = llm.chat(conversations, sampling, use_tqdm=True)
    print(f"Generated {len(outputs)} responses in {(time.perf_counter()-start)/60:.1f}m")

    out_path = gens_path(model_key, split)
    n_corr = n_trunc = n_unparse = 0
    with out_path.open("w", encoding="utf-8") as f:
        for (ex, t, seed), o in zip(meta, outputs):
            comp = o.outputs[0]
            text = comp.text.strip()
            truncated = comp.finish_reason == "length"
            parsed, status = gsm8k.extract_answer(text)
            ok = gsm8k.is_correct(parsed, ex["ground_truth"])
            n_corr += int(ok)
            n_trunc += int(truncated)
            n_unparse += int(status == "unparseable" and not truncated)
            f.write(json.dumps({
                "index": ex["index"], "question": ex["question"],
                "temperature": t, "seed": seed,
                "response": text, "truncated": truncated,
                "parse_status": status,
                "parsed": str(parsed) if parsed is not None else None,
                "ground_truth": str(ex["ground_truth"]),
                "was_correct": ok,
            }, ensure_ascii=False) + "\n")

    n = len(outputs)
    print(f"Correct     : {n_corr}/{n} ({100*n_corr/n:.1f}%)")
    print(f"Truncated   : {n_trunc} ({100*n_trunc/n:.1f}%)  "
          f"unparseable (not truncated): {n_unparse} ({100*n_unparse/n:.1f}%)")
    print(f"Wrote {out_path}")
    # Sanity band for the probe analysis: enough failures, not format chaos.
    acc = n_corr / n
    if acc > 0.93:
        print("WARNING: accuracy very high — few incorrect chains per band to probe.")
    if acc < 0.45 or n_unparse / n > 0.10:
        print("WARNING: low accuracy / high unparseable rate — check template & extraction.")


# ============================================================
# Stage 2 — extract (HF teacher-forcing, rich taps)
# ============================================================

def _common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    m = min(a.shape[-1], b.shape[-1])
    eq = (a[0, :m] == b[0, :m])
    nz = (~eq).nonzero()
    return int(nz[0].item()) if len(nz) else m


def _layer_taps(n_layers: int) -> list[int]:
    """N_LAYER_TAPS evenly spaced hidden_states indices in [1, n_layers].

    Index i of output_hidden_states is the output of layer i (0 = embeddings,
    which we skip); n_layers = the final layer (== hidden_last's source).
    """
    taps = sorted({max(1, round(i * n_layers / N_LAYER_TAPS)) for i in range(1, N_LAYER_TAPS + 1)})
    if taps[-1] != n_layers:
        taps.append(n_layers)
    return taps


@torch.no_grad()
def extract_rich(model, tokenizer, question: str, response: str, device, cfg: dict,
                 want_layers: bool) -> dict:
    """One forward pass over prompt+response -> all taps + logprob baselines."""
    prompt_msgs = gsm8k.build_messages(question)
    full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]

    prompt_ids = apply_template(tokenizer, prompt_msgs, cfg, add_generation_prompt=True)
    full_ids = apply_template(tokenizer, full_msgs, cfg, continue_final_message=True)
    prompt_len = _common_prefix_len(prompt_ids, full_ids)

    # PRE-ANSWER boundary: tokenize the response cut just before its final
    # '####' marker; the common token prefix with the full encoding is the
    # last token index that precedes the answer line. Tokenizer-agnostic.
    marker = response.rfind("####")
    has_marker = marker > 0
    if has_marker:
        pre_msgs = prompt_msgs + [{"role": "assistant", "content": response[:marker].rstrip()}]
        pre_ids = apply_template(tokenizer, pre_msgs, cfg, continue_final_message=True)
        pre_len = _common_prefix_len(pre_ids, full_ids)
        pre_len = max(prompt_len + 1, pre_len)  # never inside the prompt
    else:
        pre_len = 0  # no marker -> fall back to the end tap below

    max_total = int(cfg["max_prompt_tokens"]) + int(cfg["max_new_tokens"])
    if full_ids.shape[-1] > max_total:
        full_ids = full_ids[:, :max_total]
    full_len = int(full_ids.shape[-1])
    if not has_marker or pre_len >= full_len:
        pre_len = full_len

    full_ids = full_ids.to(device)
    out = model(input_ids=full_ids, use_cache=False, output_hidden_states=True, return_dict=True)

    hs_last = out.hidden_states[-1][0].float()   # [seq, hidden]
    resp = hs_last[prompt_len:, :]
    if resp.shape[0] == 0:
        resp = hs_last[-1:, :]
    r = resp.shape[0]

    frac_rows = [resp[min(r - 1, max(0, int(round(f * r)) - 1))] for f in FRACS]
    hidden_pre = hs_last[pre_len - 1]

    # Response-token logprobs under the model (untempered confidence baseline).
    # Row t of logits predicts token t+1: response tokens live at rows
    # [prompt_len-1, full_len-2]. log_softmax in chunks to bound memory.
    rows = out.logits[0, prompt_len - 1: full_len - 1, :]
    targets = full_ids[0, prompt_len:full_len]
    lps = []
    for i in range(0, rows.shape[0], 256):
        chunk = torch.log_softmax(rows[i:i + 256].float(), dim=-1)
        lps.append(chunk.gather(-1, targets[i:i + 256, None]).squeeze(-1))
    lp = torch.cat(lps) if lps else torch.zeros(1)
    quarters = [lp[(len(lp) * q) // 4:(len(lp) * (q + 1)) // 4] for q in range(4)]
    lp_quarters = [float(q.mean()) if len(q) else float(lp.mean()) for q in quarters]

    feat = {
        "hidden_frac": torch.stack(frac_rows).half().cpu(),   # [len(FRACS), H]
        "hidden_pre": hidden_pre.half().cpu(),                # [H]
        "has_marker": bool(has_marker and pre_len < full_len),
        "pre_frac": float((pre_len - prompt_len) / r),
        "resp_len": int(r),
        "prompt_len": int(prompt_len),
        "full_len": full_len,
        "prefix_is_clean": bool(prompt_len == prompt_ids.shape[-1]),
        "lp_mean": float(lp.mean()), "lp_min": float(lp.min()),
        "lp_sum": float(lp.sum()), "lp_quarters": lp_quarters,
    }

    if want_layers:
        _, n_layers = model_dims(model)
        taps = _layer_taps(n_layers)
        pos_idx = {
            "mid": prompt_len + min(r - 1, max(0, int(round(0.5 * r)) - 1)),
            "end": full_len - 1,
            "pre": pre_len - 1,
        }
        grid = torch.stack([
            torch.stack([out.hidden_states[li][0, pos_idx[p]].float() for p in LAYER_POSITIONS])
            for li in taps
        ])  # [n_taps, 3, H]
        feat["hidden_layers"] = grid.half().cpu()
        feat["layer_taps"] = taps

    return feat


def validate_one(model, tokenizer, rows: list[dict], device, cfg: dict, want_layers: bool) -> None:
    correct = next(r for r in rows if r["was_correct"] and not r["truncated"])
    incorrect = next(r for r in rows if not r["was_correct"] and not r["truncated"])
    H, n_layers = model_dims(model)

    for tag, row in (("CORRECT", correct), ("INCORRECT", incorrect)):
        print(f"\n=== PRE-FLIGHT {tag}  (index={row['index']}, tau={row['temperature']}) ===")
        feat = extract_rich(model, tokenizer, row["question"], row["response"], device, cfg, want_layers)
        print(f"  prompt/full len  : {feat['prompt_len']} / {feat['full_len']}  "
              f"(clean prefix: {feat['prefix_is_clean']})")
        print(f"  response tokens  : {feat['resp_len']}")
        print(f"  hidden_frac      : {tuple(feat['hidden_frac'].shape)}  expected ({len(FRACS)}, {H})")
        print(f"  pre-answer tap   : has_marker={feat['has_marker']}  pre_frac={feat['pre_frac']:.3f}")
        print(f"  logprobs         : mean={feat['lp_mean']:.3f}  min={feat['lp_min']:.2f}  "
              f"quarters={['%.2f' % q for q in feat['lp_quarters']]}")
        if want_layers:
            print(f"  layer grid       : {tuple(feat['hidden_layers'].shape)}  taps={feat['layer_taps']}"
                  f"  (model layers: {n_layers})")
        # Show where the pre-answer tap landed: decode a window around it.
        full_ids = apply_template(
            tokenizer,
            gsm8k.build_messages(row["question"]) + [{"role": "assistant", "content": row["response"]}],
            cfg, continue_final_message=True)
        pre_tok = feat["prompt_len"] + int(round(feat["pre_frac"] * feat["resp_len"])) - 1
        window = tokenizer.decode(full_ids[0, max(0, pre_tok - 15): pre_tok + 1].tolist())
        print(f"  text up to pre-answer tap: ...{window[-80:]!r}")
        assert feat["hidden_frac"].shape == (len(FRACS), H), "hidden dim mismatch"
        assert torch.isfinite(feat["hidden_frac"].float()).all(), "non-finite hidden"
        assert feat["resp_len"] > 0, "empty response region"
        if feat["has_marker"]:
            assert 0.0 < feat["pre_frac"] <= 1.0, "pre-answer tap out of range"

    print("\n  ALL ASSERTS PASSED — extraction is sane for this model.\n")


def stage_extract(model_key: str, split: str, limit: int | None,
                  want_layers: bool, do_validate: bool) -> None:
    cfg = load_config(model_key)

    now_hash = template_hash()
    print(f"Template hash: {now_hash}")

    src = gens_path(model_key, split)
    if not src.exists():
        raise FileNotFoundError(f"{src} — run --stage generate first")
    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    n = len(rows)
    print(f"Model: {model_key}  split: {split}  rows: {n}  layers: {want_layers}")

    device = get_device(cfg.get("device", "auto"))
    dtype = get_dtype(cfg["dtype_name"])
    tokenizer, model = load_model(cfg, device, dtype)
    H, n_layers = model_dims(model)
    assert H == int(cfg["hidden_size"]), f"hidden_size mismatch: model {H} vs config {cfg['hidden_size']}"

    if do_validate:
        validate_one(model, tokenizer, rows, device, cfg, want_layers)
        return

    taps = _layer_taps(n_layers)
    index = torch.empty(n, dtype=torch.long)
    temperature = torch.empty(n, dtype=torch.float32)
    was_correct = torch.empty(n, dtype=torch.bool)
    truncated = torch.empty(n, dtype=torch.bool)
    has_marker = torch.empty(n, dtype=torch.bool)
    resp_len = torch.empty(n, dtype=torch.long)
    pre_frac = torch.empty(n, dtype=torch.float32)
    lp_stats = torch.empty(n, 3, dtype=torch.float32)       # mean, min, sum
    lp_quarters = torch.empty(n, 4, dtype=torch.float32)
    hidden_frac = torch.empty(n, len(FRACS), H, dtype=torch.float16)
    hidden_pre = torch.empty(n, H, dtype=torch.float16)
    hidden_layers = (torch.empty(n, len(taps), len(LAYER_POSITIONS), H, dtype=torch.float16)
                     if want_layers else None)

    start = time.perf_counter()
    for i, row in enumerate(rows):
        feat = extract_rich(model, tokenizer, row["question"], row["response"],
                            device, cfg, want_layers)
        index[i] = row["index"]
        temperature[i] = row["temperature"]
        was_correct[i] = row["was_correct"]
        truncated[i] = row["truncated"]
        has_marker[i] = feat["has_marker"]
        resp_len[i] = feat["resp_len"]
        pre_frac[i] = feat["pre_frac"]
        lp_stats[i] = torch.tensor([feat["lp_mean"], feat["lp_min"], feat["lp_sum"]])
        lp_quarters[i] = torch.tensor(feat["lp_quarters"])
        hidden_frac[i] = feat["hidden_frac"]
        hidden_pre[i] = feat["hidden_pre"]
        if want_layers:
            hidden_layers[i] = feat["hidden_layers"]
        if (i + 1) % 200 == 0 or (i + 1) == n:
            el = time.perf_counter() - start
            rate = (i + 1) / el
            print(f"  {i+1}/{n}  ({rate:.1f}/s, eta {(n-(i+1))/rate/60:.1f}m)", flush=True)

    payload = {
        "index": index, "temperature": temperature, "was_correct": was_correct,
        "truncated": truncated, "has_marker": has_marker,
        "resp_len": resp_len, "pre_frac": pre_frac,
        "lp_stats": lp_stats, "lp_quarters": lp_quarters,
        "hidden_frac": hidden_frac, "fracs": FRACS,
        "hidden_pre": hidden_pre,
        "meta": {"model_key": model_key, "model": cfg["model_name_or_path"],
                 "split": split, "n": n, "hidden_size": H, "n_layers": n_layers,
                 "fracs": FRACS, "temps": TEMPS, "dtype": cfg["dtype_name"],
                 "template_hash": now_hash,
                 "lp_stats_cols": ["mean", "min", "sum"],
                 "note": "clean generations, rich taps (fracs + pre-answer + logprobs)"},
    }
    if want_layers:
        payload["hidden_layers"] = hidden_layers
        payload["meta"]["layer_taps"] = taps
        payload["meta"]["layer_positions"] = LAYER_POSITIONS

    out_path = pt_path(model_key, split)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    print(f"\nDone in {(time.perf_counter()-start)/60:.1f}m. Saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ministral", "qwen", "llama"])
    ap.add_argument("--stage", required=True, choices=["generate", "extract"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None, help="cap prompts (generate) / rows (extract)")
    ap.add_argument("--layers", action="store_true",
                    help="extract: also store the depth sweep (test split only; big)")
    ap.add_argument("--validate-one", action="store_true",
                    help="extract: pre-flight one correct + one incorrect row, assert, no save")
    args = ap.parse_args()

    if args.stage == "generate":
        stage_generate(args.model, args.split, args.limit)
    else:
        stage_extract(args.model, args.split, args.limit, args.layers, args.validate_one)


if __name__ == "__main__":
    main()
