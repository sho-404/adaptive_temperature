"""Confound controls + calibration for the end-of-reasoning correctness signal.

All within temperature bands, GroupKFold-by-prompt (as geometry.py):

  C1  LENGTH CONTROL. Held-out AUC of response length alone vs the hidden
      state. If hidden >> length, the probe is not just reading "long
      solutions fail". Also: Spearman correlation between the arrow score
      and length (is the dial secretly a word-counter?).

  C2  CALIBRATION. Held-out probabilities of the fitted probe -> ECE
      (10 equal-width bins) and Brier score. A calibrated dial supports
      the claim "the model knows (too late)" rather than merely
      "a classifier can be trained".

Runs locally off data/gsm8k_en_reasoning.pt. Writes paper/controls.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "controls.json"

MIN_PER_CLASS = 30
N_SPLITS = 5
ECE_BINS = 10

R = torch.load(DATA / "gsm8k_en_reasoning.pt", map_location="cpu", weights_only=False)
y_all = R["was_correct"].numpy().astype(int)
t_all = R["temperature"].numpy()
g_all = R["index"].numpy()
rlen_all = R["resp_len"].numpy().astype(float)
fracs = [float(f) for f in R["fracs"]]
H_end = R["hidden_frac"].numpy()[:, fracs.index(1.0), :]


def heldout_proba(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Held-out probe probabilities via GroupKFold (every row scored by a model
    that never saw its prompt)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    p = np.zeros(len(y))
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def heldout_arrow_score(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    s = np.zeros(len(y))
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        w = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        s[te] = X[te] @ w
    return s


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


results: dict = {"bands": {}}

for tv in sorted(set(t_all.tolist())):
    m = t_all == tv
    y, g, rl, X = y_all[m], g_all[m], rlen_all[m], H_end[m]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if min(n_pos, n_neg) < MIN_PER_CLASS:
        print(f"tau={tv:.1f}: skipped (correct={n_pos}, incorrect={n_neg})")
        continue

    band: dict = {"n": int(m.sum()), "correct": n_pos, "incorrect": n_neg}

    # C1 — length control (hidden+len tests whether hidden subsumes the length signal)
    p_len = heldout_proba(rl.reshape(-1, 1), y, g)
    p_hid = heldout_proba(X, y, g)
    p_both = heldout_proba(np.hstack([X, rl.reshape(-1, 1)]), y, g)
    band["len_auc"] = float(roc_auc_score(y, p_len))
    band["hidden_auc"] = float(roc_auc_score(y, p_hid))
    band["hidden_plus_len_auc"] = float(roc_auc_score(y, p_both))
    arrow = heldout_arrow_score(X, y, g)
    band["spearman_arrow_len"] = float(spearmanr(arrow, rl).statistic)
    band["spearman_len_correct"] = float(spearmanr(rl, y).statistic)

    # C2 — calibration of the hidden-state probe
    band["ece"] = ece(y, p_hid)
    band["brier"] = float(brier_score_loss(y, p_hid))
    band["base_rate"] = float(y.mean())

    results["bands"][f"{tv:.1f}"] = band
    print(f"tau={tv:.1f}  (n={band['n']}, base_rate={band['base_rate']:.2f})")
    print(f"  C1: length-only AUC={band['len_auc']:.3f}   hidden AUC={band['hidden_auc']:.3f}"
          f"   hidden+len AUC={band['hidden_plus_len_auc']:.3f}"
          f"   spearman(arrow,len)={band['spearman_arrow_len']:+.2f}"
          f"   spearman(len,correct)={band['spearman_len_correct']:+.2f}")
    print(f"  C2: ECE={band['ece']:.3f}   Brier={band['brier']:.3f}")

if results["bands"]:
    bs = results["bands"].values()
    results["macro"] = {k: float(np.mean([b[k] for b in bs]))
                        for k in ("len_auc", "hidden_auc", "hidden_plus_len_auc",
                                  "spearman_arrow_len", "ece", "brier")}
    mm = results["macro"]
    print("\n=== MACRO (mean over temperature bands) ===")
    print(f"  length-only AUC={mm['len_auc']:.3f}   hidden AUC={mm['hidden_auc']:.3f}"
          f"   spearman(arrow,len)={mm['spearman_arrow_len']:+.2f}")
    print(f"  ECE={mm['ece']:.3f}   Brier={mm['brier']:.3f}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT.relative_to(ROOT)}")
