"""Positional probe: how EARLY in the reasoning does correctness become predictable?

For each reasoning fraction (25/50/75/100%), predict was_correct from the hidden
state at that point, WITHIN each temperature band (confound removed). If AUC is
already high by ~50%, a token-level decoder could act on it mid-generation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).parents[2] / "data"
R = torch.load(DATA / "gsm8k_en_reasoning.pt", map_location="cpu", weights_only=False)

y = R["was_correct"].numpy().astype(int)
t = R["temperature"].numpy()
fracs = R["fracs"]
Hf = R["hidden_frac"].numpy()            # [N, F, hidden]

def auc_cv(X, yy):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return roc_auc_score(yy, cross_val_predict(clf, X, yy, cv=cv, method="predict_proba")[:, 1])

bands = [tv for tv in sorted(set(t.tolist()))]
print("=" * 64)
print("WITHIN-TEMPERATURE AUC of hidden -> was_correct, by reasoning fraction")
print("(how early the model 'knows' it is on track)")
print("=" * 64)
header = "  temp     n  %corr  " + "  ".join(f"{int(f*100):>4d}%" for f in fracs)
print(header)
for tv in bands:
    m = (t == tv); n = int(m.sum()); pc = y[m].mean()
    if min(int((y[m] == 1).sum()), int((y[m] == 0).sum())) < 60:
        print(f"  {tv:4.1f} {n:6d} {pc:6.2f}   (too few of a class)"); continue
    cells = "  ".join(f"{auc_cv(Hf[m, j, :], y[m]):.3f}" for j in range(len(fracs)))
    print(f"  {tv:4.1f} {n:6d} {pc:6.2f}   {cells}")
print("=" * 64)
