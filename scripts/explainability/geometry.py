"""Geometry of the correctness signal in reasoning hidden states.

Three questions, all within temperature bands (confound removed), all with
GroupKFold-by-prompt CV (a prompt's two attempts never split across folds):

  Q1  THE ARROW. Is a single difference-of-means direction (mean of correct
      hiddens minus mean of incorrect hiddens, end of reasoning) as good a
      correctness reader as a fitted logistic-regression probe?
      If yes -> the signal is a linear "dial" the model itself could read
      with one dot product (World A).

  Q2  POSITION TRANSFER. Take the arrow fitted at the END of the reasoning
      (frac 1.0) and apply it, unchanged, to hidden states at 25/50/75%.
      Grows-vs-snaps: if the same arrow already separates mid-generation,
      it is ONE representation filling up over time; if it reads nothing
      mid-stream, the end-state representation is a different thing that
      only crystallises once the answer is written.
      Reference: the "native" arrow fitted AT each fraction.

  Q3  ARROW ALIGNMENT. Cosine similarity between per-fraction arrows.
      High cosine across fractions = same axis throughout (the dial);
      low-to-high drift = the representation rotates into place late.

Runs locally off data/gsm8k_en_reasoning.pt. Writes paper/geometry.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "geometry.json"

MIN_PER_CLASS = 30  # skip a temperature band if either class is thinner than this
N_SPLITS = 5

R = torch.load(DATA / "gsm8k_en_reasoning.pt", map_location="cpu", weights_only=False)
y_all = R["was_correct"].numpy().astype(int)
t_all = R["temperature"].numpy()
g_all = R["index"].numpy()  # group by prompt: both attempts stay in one fold
fracs = [float(f) for f in R["fracs"]]
Hf = R["hidden_frac"].numpy()  # [N, F, hidden]
F_END = fracs.index(1.0)


def diffmean_auc(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """Held-out AUC of the difference-of-means arrow (no fitting beyond two means)."""
    aucs = []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        w = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        aucs.append(roc_auc_score(y[te], X[te] @ w))
    return float(np.mean(aucs))


def logreg_auc(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """Held-out AUC of the fitted probe (the ceiling the arrow is compared against)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    aucs = []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def transfer_auc(X_end: np.ndarray, X_mid: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """Arrow fitted on END states (train folds), applied unchanged to MID states (test folds)."""
    aucs = []
    for tr, te in GroupKFold(N_SPLITS).split(X_end, y, g):
        w = X_end[tr][y[tr] == 1].mean(0) - X_end[tr][y[tr] == 0].mean(0)
        aucs.append(roc_auc_score(y[te], X_mid[te] @ w))
    return float(np.mean(aucs))


def full_arrow(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / np.linalg.norm(w)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


results: dict = {"fracs": fracs, "bands": {}}
bands = sorted(set(t_all.tolist()))

for tv in bands:
    m = t_all == tv
    y, g = y_all[m], g_all[m]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if min(n_pos, n_neg) < MIN_PER_CLASS:
        print(f"tau={tv:.1f}: skipped (correct={n_pos}, incorrect={n_neg})")
        continue

    X_end = Hf[m, F_END, :]
    band: dict = {"n": int(m.sum()), "correct": n_pos, "incorrect": n_neg}

    # Q1 — the arrow vs the fitted probe, at the end of reasoning
    band["arrow_auc_end"] = diffmean_auc(X_end, y, g)
    band["probe_auc_end"] = logreg_auc(X_end, y, g)

    # Q2 — end-arrow applied to earlier positions vs the native per-position arrow
    band["transfer"] = {}
    for fi, fv in enumerate(fracs):
        X_mid = Hf[m, fi, :]
        band["transfer"][f"{fv:.2f}"] = {
            "end_arrow_auc": transfer_auc(X_end, X_mid, y, g),
            "native_arrow_auc": diffmean_auc(X_mid, y, g),
        }

    # Q3 — alignment of per-fraction arrows (fitted on the full band; descriptive)
    ws = [full_arrow(Hf[m, fi, :], y) for fi in range(len(fracs))]
    band["cos_to_end"] = {f"{fracs[fi]:.2f}": cos(ws[fi], ws[F_END]) for fi in range(len(fracs))}

    results["bands"][f"{tv:.1f}"] = band

    print(f"\ntau={tv:.1f}  (n={band['n']}, correct={n_pos}, incorrect={n_neg})")
    print(f"  Q1 end-of-reasoning : arrow AUC={band['arrow_auc_end']:.3f}  vs  probe AUC={band['probe_auc_end']:.3f}")
    print("  Q2 transfer (end-arrow -> position) vs native arrow:")
    for fv in fracs:
        tr_ = band["transfer"][f"{fv:.2f}"]
        print(f"     {int(fv*100):3d}%: end-arrow={tr_['end_arrow_auc']:.3f}  native={tr_['native_arrow_auc']:.3f}")
    print("  Q3 cosine(arrow@frac, arrow@end): "
          + "  ".join(f"{int(fv*100)}%={band['cos_to_end'][f'{fv:.2f}']:.2f}" for fv in fracs))

# macro summary across usable bands
if results["bands"]:
    def macro(fn):
        return float(np.mean([fn(b) for b in results["bands"].values()]))
    results["macro"] = {
        "arrow_auc_end": macro(lambda b: b["arrow_auc_end"]),
        "probe_auc_end": macro(lambda b: b["probe_auc_end"]),
        "transfer_end_arrow": {f"{fv:.2f}": macro(lambda b, fv=fv: b["transfer"][f"{fv:.2f}"]["end_arrow_auc"]) for fv in fracs},
        "transfer_native": {f"{fv:.2f}": macro(lambda b, fv=fv: b["transfer"][f"{fv:.2f}"]["native_arrow_auc"]) for fv in fracs},
        "cos_to_end": {f"{fv:.2f}": macro(lambda b, fv=fv: b["cos_to_end"][f"{fv:.2f}"]) for fv in fracs},
    }
    mm = results["macro"]
    print("\n=== MACRO (mean over temperature bands) ===")
    print(f"  Q1 arrow={mm['arrow_auc_end']:.3f}  probe={mm['probe_auc_end']:.3f}")
    for fv in fracs:
        print(f"  Q2 {int(fv*100):3d}%: end-arrow={mm['transfer_end_arrow'][f'{fv:.2f}']:.3f}"
              f"  native={mm['transfer_native'][f'{fv:.2f}']:.3f}"
              f"  cos={mm['cos_to_end'][f'{fv:.2f}']:.2f}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT.relative_to(ROOT)}")
