"""CLEAN-SPLIT VALIDATION of the end-of-reasoning correctness signal.

All headline numbers so far come from the LPO-pair data: train-split prompts,
responses picked by the pair-generation logic (selection skew). This script
re-validates on data with NO selection at all: fresh solutions sampled on the
GSM8K TEST split, scored against ground truth, with the probes/arrows fitted
on the pair data applied FROZEN.

Stages (run in order; generate/extract on the GPU box, eval on the Mac):

  --stage generate   [VM, vLLM]  sample solutions on the test split at
                     TEMPS x K, score them -> data/gsm8k_en_clean_gens.jsonl
  --stage extract    [VM, HF eager]  teacher-force each generation, tap the
                     last-layer hidden at 25/50/75/100% of the response
                     -> data/gsm8k_en_clean.pt
  --stage eval       [Mac]  fit arrow + logreg probe per temperature band on
                     the FULL pair data (train), apply frozen to the clean
                     test rows -> paper/clean_validation.json

Guardrail: the extract stage asserts the prompt-template hash matches the one
stamped into the pair-era features, so train/test features are byte-identical
in formatting.
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

DATA = Path(__file__).parents[2] / "data"
OUT_GENS = DATA / "gsm8k_en_clean_gens.jsonl"
OUT_PT = DATA / "gsm8k_en_clean.pt"
OUT_JSON = Path(__file__).parents[2] / "paper" / "clean_validation.json"

TEMPS = [(0.0, 1), (0.6, 2), (1.0, 2)]  # (temperature, samples per prompt)
MIN_PER_CLASS = 30


# ============================================================
# Stage 1 — generate (vLLM, batched across ALL prompts)
# ============================================================

def stage_generate(limit: int | None) -> None:
    from model import load_config, load_vllm
    import tasks.gsm8k as gsm8k
    from vllm import SamplingParams

    cfg = load_config()
    examples = gsm8k.load("en", "test")
    if limit:
        examples = examples[:limit]
    print(f"Test prompts: {len(examples)}  |  temps: {TEMPS}")

    # One flat batch: every (prompt, temperature, sample) is one request.
    conversations, meta = [], []
    for ex in examples:
        msgs = gsm8k.build_messages(ex["question"])
        for ti, (t, k) in enumerate(TEMPS):
            for s in range(k):
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
    el = time.perf_counter() - start
    print(f"Generated {len(outputs)} responses in {el/60:.1f}m")

    n_corr = 0
    with OUT_GENS.open("w", encoding="utf-8") as f:
        for (ex, t, seed), o in zip(meta, outputs):
            comp = o.outputs[0]
            text = comp.text.strip()
            truncated = comp.finish_reason == "length"
            parsed, status = gsm8k.extract_answer(text)
            ok = gsm8k.is_correct(parsed, ex["ground_truth"])
            n_corr += int(ok)
            f.write(json.dumps({
                "index": ex["index"], "question": ex["question"],
                "temperature": t, "seed": seed,
                "response": text, "truncated": truncated,
                "parse_status": status,
                "parsed": str(parsed) if parsed is not None else None,
                "ground_truth": str(ex["ground_truth"]),
                "was_correct": ok,
            }, ensure_ascii=False) + "\n")
    print(f"Correct: {n_corr}/{len(outputs)} ({100*n_corr/len(outputs):.1f}%)")
    print(f"Wrote {OUT_GENS}")


# ============================================================
# Stage 2 — extract (HF eager teacher-forcing, reuses pair-era code)
# ============================================================

def stage_extract(limit: int | None) -> None:
    from model import load_config, get_device, get_dtype, load_model
    from extract_features import template_hash
    from extract_reasoning_features import extract, FRACS

    cfg = load_config()

    # Guardrail: identical prompt formatting to the pair-era features.
    pair_feats = torch.load(DATA / "gsm8k_en.pt", map_location="cpu", weights_only=False)
    pair_hash = pair_feats["meta"]["template_hash"]
    now_hash = template_hash()
    assert now_hash == pair_hash, (
        f"template hash mismatch: pair data {pair_hash} vs current {now_hash} — "
        "prompt formatting drifted; frozen probes would not be comparable."
    )
    del pair_feats
    print(f"Template hash OK ({now_hash})")

    rows = [json.loads(l) for l in OUT_GENS.open(encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    n = len(rows)
    print(f"Rows to extract: {n}")

    device = get_device(cfg.get("device", "auto"))
    dtype = get_dtype(cfg["dtype_name"])
    tokenizer, model = load_model(cfg, device, dtype)

    H = int(cfg["hidden_size"])
    index = torch.empty(n, dtype=torch.long)
    temperature = torch.empty(n, dtype=torch.float32)
    was_correct = torch.empty(n, dtype=torch.bool)
    truncated = torch.empty(n, dtype=torch.bool)
    resp_len = torch.empty(n, dtype=torch.long)
    hidden_frac = torch.empty(n, len(FRACS), H, dtype=torch.float32)

    start = time.perf_counter()
    for i, r in enumerate(rows):
        feat = extract(model, tokenizer, r["question"], r["response"], device, cfg)
        index[i] = r["index"]
        temperature[i] = r["temperature"]
        was_correct[i] = r["was_correct"]
        truncated[i] = r["truncated"]
        resp_len[i] = feat["resp_len"]
        hidden_frac[i] = feat["hidden_frac"]
        if (i + 1) % 200 == 0 or (i + 1) == n:
            el = time.perf_counter() - start
            rate = (i + 1) / el
            print(f"  {i+1}/{n}  ({rate:.1f}/s, eta {(n-(i+1))/rate/60:.1f}m)", flush=True)

    payload = {
        "index": index, "temperature": temperature, "was_correct": was_correct,
        "truncated": truncated, "resp_len": resp_len,
        "hidden_frac": hidden_frac, "fracs": FRACS,
        "meta": {"split": "test", "n": n, "hidden_size": H,
                 "model": cfg["model_name_or_path"], "dtype": cfg["dtype_name"],
                 "template_hash": now_hash, "temps": TEMPS,
                 "note": "clean validation: fresh test-split generations, no selection"},
    }
    tmp = OUT_PT.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(OUT_PT)
    print(f"\nDone in {(time.perf_counter()-start)/60:.1f}m. Saved -> {OUT_PT}")


# ============================================================
# Stage 3 — eval (frozen probes from pair data -> clean test rows)
# ============================================================

def stage_eval() -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def arrow_cv(X, y, g):
        a = []
        for tr, te in GroupKFold(5).split(X, y, g):
            w_ = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
            a.append(roc_auc_score(y[te], X[te] @ w_))
        return float(np.mean(a))

    def probe_cv(X, y, g):
        clf_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        a = []
        for tr, te in GroupKFold(5).split(X, y, g):
            clf_.fit(X[tr], y[tr])
            a.append(roc_auc_score(y[te], clf_.predict_proba(X[te])[:, 1]))
        return float(np.mean(a))

    P = torch.load(DATA / "gsm8k_en_reasoning.pt", map_location="cpu", weights_only=False)  # pair (train)
    C = torch.load(OUT_PT, map_location="cpu", weights_only=False)                          # clean (test)
    assert P["meta"]["template_hash"] == C["meta"]["template_hash"], "template hash mismatch"

    fracs = [float(f) for f in P["fracs"]]
    F_END = fracs.index(1.0)

    yp, tp = P["was_correct"].numpy().astype(int), P["temperature"].numpy()
    Hp = P["hidden_frac"].numpy()
    yc, tc = C["was_correct"].numpy().astype(int), C["temperature"].numpy()
    Hc = C["hidden_frac"].numpy()
    gc = C["index"].numpy()
    keep = ~C["truncated"].numpy()
    print(f"Clean rows: {len(yc)}  (dropping {int((~keep).sum())} truncated)")
    yc, tc, Hc, gc = yc[keep], tc[keep], Hc[keep], gc[keep]

    results: dict = {"fracs": fracs, "bands": {}}
    for tv in sorted(set(tc.tolist())):
        mp, mc = tp == tv, tc == tv
        y_tr, y_te = yp[mp], yc[mc]
        if min(y_tr.sum(), (1 - y_tr).sum()) < MIN_PER_CLASS or \
           min(y_te.sum(), (1 - y_te).sum()) < MIN_PER_CLASS:
            print(f"tau={tv:.1f}: skipped (train {int(y_tr.sum())}/{int((1-y_tr).sum())}, "
                  f"test {int(y_te.sum())}/{int((1-y_te).sum())})")
            continue

        X_tr = Hp[mp, F_END, :]
        # frozen arrow + frozen probe, fitted once on the FULL pair band
        w = X_tr[y_tr == 1].mean(0) - X_tr[y_tr == 0].mean(0)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(X_tr, y_tr)

        band = {"n_test": int(mc.sum()), "test_correct": int(y_te.sum()),
                "test_base_rate": float(y_te.mean()),
                "arrow_auc": float(roc_auc_score(y_te, Hc[mc, F_END, :] @ w)),
                "probe_auc": float(roc_auc_score(y_te, clf.predict_proba(Hc[mc, F_END, :])[:, 1])),
                "positions": {}}
        for fi, fv in enumerate(fracs):  # end-arrow applied to earlier clean positions
            band["positions"][f"{fv:.2f}"] = float(roc_auc_score(y_te, Hc[mc, fi, :] @ w))

        # refit WITHIN the clean band (CV): the clean-data signal ceiling, to
        # separate "signal is weaker on clean data" from "transfer gap".
        Xc, y_, g_ = Hc[mc], y_te, gc[mc]
        band["refit"] = {
            "arrow_auc": arrow_cv(Xc[:, F_END, :], y_, g_),
            "probe_auc": probe_cv(Xc[:, F_END, :], y_, g_),
            "positions": {f"{fv:.2f}": arrow_cv(Xc[:, fi, :], y_, g_)
                          for fi, fv in enumerate(fracs)},
        }

        results["bands"][f"{tv:.1f}"] = band
        pos = "  ".join(f"{int(fv*100)}%={band['positions'][f'{fv:.2f}']:.3f}" for fv in fracs)
        rpos = "  ".join(f"{int(fv*100)}%={band['refit']['positions'][f'{fv:.2f}']:.3f}" for fv in fracs)
        print(f"tau={tv:.1f}  n_test={band['n_test']}  base_rate={band['test_base_rate']:.2f}")
        print(f"  FROZEN end-of-reasoning: arrow AUC={band['arrow_auc']:.3f}  probe AUC={band['probe_auc']:.3f}")
        print(f"  end-arrow by position  : {pos}")
        print(f"  REFIT (clean ceiling)  : arrow={band['refit']['arrow_auc']:.3f}  probe={band['refit']['probe_auc']:.3f}")
        print(f"  refit arrow by position: {rpos}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["generate", "extract", "eval"])
    ap.add_argument("--limit", type=int, default=None, help="cap prompts (generate) / rows (extract)")
    args = ap.parse_args()
    {"generate": lambda: stage_generate(args.limit),
     "extract": lambda: stage_extract(args.limit),
     "eval": stage_eval}[args.stage]()


if __name__ == "__main__":
    main()
