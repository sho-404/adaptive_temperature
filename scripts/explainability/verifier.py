"""VERIFIER DEMO: is the (late) arrow score useful, not just readable?

Each test prompt has ~5 chains (1 @ tau=0, 2 @ 0.6, 2 @ 1.0). The
temperature-invariant end-of-chain arrow scores every chain; we compare
answer-selection strategies:

  greedy        answer of the tau=0 chain (the no-sampling baseline)
  majority      majority vote over parsed answers (ties -> greedy's answer if
                tied, else first-seen; unparseable chains abstain)
  best_of_5     answer of the single highest-arrow-score chain
  arrow_vote    weighted vote: each chain votes with sigmoid(standardized
                arrow score) — standardization fitted on the TRAIN split
  single_avg    mean accuracy of one sampled chain (expected 1-sample value)
  oracle        upper bound: correct if ANY chain is correct

Also RISK-COVERAGE for best_of_5: use the winning chain's arrow score as
confidence; accuracy among the top X% most-confident prompts.

Prompts whose chains are all truncated are skipped; unparseable answers count
as wrong for the strategy that picks them. Writes paper/verifier.json.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "verifier.json"

MODELS = ["ministral", "qwen", "llama"]
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.5]


def load(model: str):
    d = torch.load(DATA / f"gsm8k_en_clean.{model}.test.pt", map_location="cpu", weights_only=False)
    fracs = [float(f) for f in d["fracs"]]
    X = d["hidden_frac"][:, fracs.index(1.0), :].numpy().astype(np.float32)
    rows = [json.loads(l) for l in
            (DATA / f"gsm8k_en_clean_gens.{model}.test.jsonl").open(encoding="utf-8") if l.strip()]
    assert len(rows) == len(X), "pt/jsonl row mismatch"
    for r, idx, tv in zip(rows, d["index"].numpy(), d["temperature"].numpy()):
        assert r["index"] == int(idx) and abs(r["temperature"] - float(tv)) < 1e-6, "row alignment broke"
    tr = torch.load(DATA / f"gsm8k_en_clean.{model}.train.pt", map_location="cpu", weights_only=False)
    Xtr = tr["hidden_frac"][:, fracs.index(1.0), :].numpy().astype(np.float32)
    ytr = tr["was_correct"].numpy().astype(int)
    w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    s_tr = Xtr @ w
    mu, sd = float(s_tr.mean()), float(s_tr.std())
    return X @ w, rows, d["truncated"].numpy(), mu, sd


def run_model(model: str) -> dict:
    scores, rows, trunc, mu, sd = load(model)
    by_prompt = defaultdict(list)
    for s, r, tr in zip(scores, rows, trunc):
        if not tr:
            by_prompt[r["index"]].append(
                (float(s), r["parsed"], bool(r["was_correct"]), float(r["temperature"])))

    acc = Counter()
    n = 0
    conf_correct = []  # (confidence, best_of_5 correct) for risk-coverage
    for chains in by_prompt.values():
        n += 1
        greedy = next((c for c in chains if c[3] == 0.0), None)
        acc["greedy"] += bool(greedy and greedy[2])
        acc["single_avg"] += float(np.mean([c[2] for c in chains]))
        acc["oracle"] += any(c[2] for c in chains)

        best = max(chains, key=lambda c: c[0])
        acc["best_of_5"] += best[2]
        conf_correct.append((best[0], best[2]))

        voted = [c for c in chains if c[1] is not None]
        if voted:
            counts = Counter(c[1] for c in voted)
            top = max(counts.values())
            tied = {a for a, k in counts.items() if k == top}
            if len(tied) > 1 and greedy and greedy[1] in tied:
                pick = greedy[1]
            else:
                pick = next(c[1] for c in voted if c[1] in tied)
            acc["majority"] += next(c[2] for c in voted if c[1] == pick)

            weights = defaultdict(float)
            for s, a, _, _ in voted:
                weights[a] += 1.0 / (1.0 + np.exp(-(s - mu) / sd))
            wpick = max(weights, key=weights.get)
            acc["arrow_vote"] += next(c[2] for c in voted if c[1] == wpick)

    res = {k: acc[k] / n for k in ("greedy", "single_avg", "majority",
                                   "best_of_5", "arrow_vote", "oracle")}
    res["n_prompts"] = n

    conf_correct.sort(key=lambda x: -x[0])
    res["risk_coverage_best_of_5"] = {
        f"{c:.0%}": float(np.mean([ok for _, ok in conf_correct[: max(1, int(c * n))]]))
        for c in COVERAGES
    }
    return res


results = {}
for m in MODELS:
    results[m] = run_model(m)
    r = results[m]
    print(f"{m} (n={r['n_prompts']}):")
    print(f"  greedy={r['greedy']:.3f}  single_avg={r['single_avg']:.3f}  "
          f"majority={r['majority']:.3f}  best_of_5={r['best_of_5']:.3f}  "
          f"arrow_vote={r['arrow_vote']:.3f}  oracle={r['oracle']:.3f}")
    rc = r["risk_coverage_best_of_5"]
    print("  risk-coverage (best_of_5): " + "  ".join(f"{k}:{v:.3f}" for k, v in rc.items()))

OUT.write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT.relative_to(ROOT)}")
