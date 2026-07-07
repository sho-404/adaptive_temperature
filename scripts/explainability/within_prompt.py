"""WITHIN-PROMPT test: does the arrow read the ATTEMPT or the PROBLEM?

Pooled AUC can be inflated by problem difficulty (hard prompts -> mostly
incorrect chains AND distinctive end-of-chain states). Here the prompt is held
fixed: among the ~5 test chains OF THE SAME PROBLEM, does the arrow score the
correct attempts above the incorrect ones? Difficulty is constant within a
prompt, so whatever survives is pure chain-level ("attempt") signal.

Metric: within-prompt pairwise accuracy = P(score(correct) > score(incorrect))
over all (correct, incorrect) chain pairs drawn from the SAME prompt — i.e. a
Mann-Whitney/AUC restricted to same-prompt pairs. Two variants:
  same_band  — both chains sampled at the SAME temperature (cleanest control;
               only tau in {0.6, 1.0} yield same-band pairs, k=2 per band)
  mixed      — any pair among the prompt's 5 chains (the practical
               best-of-5-verifier setting; crosses temperatures)
Reference: the pooled cross-prompt AUC on the identical rows.

Arrow: diff-of-means on the train split, pooled over bands (band arrows are
nearly collinear, cos 0.93-0.99). 95% CIs by bootstrap over prompts.

Runs locally. Writes paper/within_prompt.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "within_prompt.json"

MODELS = ["ministral", "qwen", "llama"]
B = 1000
RNG = np.random.default_rng(0)


def load(model: str, split: str):
    d = torch.load(DATA / f"gsm8k_en_clean.{model}.{split}.pt",
                   map_location="cpu", weights_only=False)
    keep = ~d["truncated"].numpy()
    fracs = [float(f) for f in d["fracs"]]
    X = d["hidden_frac"][:, fracs.index(1.0), :].numpy().astype(np.float32)[keep]
    return (X, d["was_correct"].numpy().astype(int)[keep],
            d["temperature"].numpy()[keep], d["index"].numpy()[keep])


def pairs_stats(items) -> tuple[int, int]:
    """(#correctly ordered pairs, #pairs) over (pos, neg) products, ties = 0.5
    handled by counting twice (win=2, tie=1, scale by 2)."""
    wins = tot = 0
    pos = [s for s, y in items if y == 1]
    neg = [s for s, y in items if y == 0]
    for sp in pos:
        for sn in neg:
            tot += 2
            wins += 2 if sp > sn else (1 if sp == sn else 0)
    return wins, tot


def within_prompt_acc(scores, y, t, g, same_band: bool):
    """Per-prompt pairwise stats -> (acc, per-prompt list for bootstrap)."""
    by_prompt = defaultdict(list)
    for s, yy, tt, gg in zip(scores, y, t, g):
        by_prompt[gg].append((s, yy, tt))
    per_prompt = {}
    for gg, chains in by_prompt.items():
        if same_band:
            w = n = 0
            for tv in {0.6, 1.0}:
                ww, nn = pairs_stats([(s, yy) for s, yy, tt in chains if tt == tv])
                w += ww; n += nn
        else:
            w, n = pairs_stats([(s, yy) for s, yy, tt in chains])
        if n:
            per_prompt[gg] = (w, n)
    wins = sum(w for w, _ in per_prompt.values())
    tot = sum(n for _, n in per_prompt.values())
    return wins / tot, per_prompt


def boot_ci(per_prompt: dict) -> list[float]:
    keys = np.array(list(per_prompt.keys()))
    w = np.array([per_prompt[k][0] for k in keys], dtype=float)
    n = np.array([per_prompt[k][1] for k in keys], dtype=float)
    accs = []
    for _ in range(B):
        idx = RNG.integers(0, len(keys), len(keys))
        if n[idx].sum():
            accs.append(w[idx].sum() / n[idx].sum())
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return [float(lo), float(hi)]


results = {}
for model in MODELS:
    Xtr, ytr, _, _ = load(model, "train")
    Xte, yte, tte, gte = load(model, "test")
    w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)   # pooled-band arrow
    s = Xte @ w

    pooled = float(roc_auc_score(yte, s))
    acc_sb, pp_sb = within_prompt_acc(s, yte, tte, gte, same_band=True)
    acc_mx, pp_mx = within_prompt_acc(s, yte, tte, gte, same_band=False)

    results[model] = {
        "pooled_auc": pooled,
        "within_same_band": {"acc": acc_sb, "ci": boot_ci(pp_sb),
                             "n_prompts": len(pp_sb),
                             "n_pairs": int(sum(n for _, n in pp_sb.values()) // 2)},
        "within_mixed": {"acc": acc_mx, "ci": boot_ci(pp_mx),
                         "n_prompts": len(pp_mx),
                         "n_pairs": int(sum(n for _, n in pp_mx.values()) // 2)},
    }
    r = results[model]
    print(f"{model}: pooled AUC={pooled:.3f}")
    print(f"  within-prompt SAME-BAND: {acc_sb:.3f} {[round(x,3) for x in r['within_same_band']['ci']]} "
          f"({r['within_same_band']['n_prompts']} prompts, {r['within_same_band']['n_pairs']} pairs)")
    print(f"  within-prompt MIXED    : {acc_mx:.3f} {[round(x,3) for x in r['within_mixed']['ci']]} "
          f"({r['within_mixed']['n_prompts']} prompts, {r['within_mixed']['n_pairs']} pairs)")

OUT.write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT.relative_to(ROOT)}")
