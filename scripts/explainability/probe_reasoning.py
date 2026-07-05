"""ORACLE PROBE analysis: can features OVER THE REASONING predict correctness?

Compares, with GroupKFold-by-prompt 5-fold CV (a prompt's two attempts never
split across folds), the AUC of predicting `was_correct` from:
  - temperature only           (confound control)
  - response length only       (confound control)
  - temp + length              (confounds combined)
  - PROMPT-ONLY hidden         (structural baseline: same prompt for both
                                attempts -> must be ~0.5)
  - reasoning hidden_mean      (pooled over the reasoning)
  - reasoning hidden_last      (end of reasoning)
  - hidden_last + confounds    (does reasoning add over the trivial controls?)

If the reasoning hiddens beat the controls and the ~0.59 prompt ceiling, the
signal lives in the reasoning -> token-level is worth building.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = Path(__file__).parents[2] / "data"
LANG = "en"

R = torch.load(DATA / f"gsm8k_{LANG}_reasoning.pt", map_location="cpu", weights_only=False)
P = torch.load(DATA / f"gsm8k_{LANG}.pt", map_location="cpu", weights_only=False)

y = R["was_correct"].numpy().astype(int)
groups = R["index"].numpy()                       # group by prompt
temp = R["temperature"].numpy().reshape(-1, 1)
rlen = R["resp_len"].numpy().reshape(-1, 1).astype(float)
h_last = R["hidden_last"].numpy()
h_mean = R["hidden_mean"].numpy()

# Prompt-only hidden, aligned to each reasoning row by its prompt index.
p_row = {int(i): k for k, i in enumerate(P["index"].tolist())}
p_hidden = np.stack([P["hidden"][p_row[int(i)]].numpy() for i in R["index"].tolist()])

cv = GroupKFold(n_splits=5)

def auc(X, name):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    pr = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    print(f"  {name:34s} AUC={roc_auc_score(y, pr):.4f}  AP={average_precision_score(y, pr):.4f}")

print("=" * 60)
print(f"Predicting was_correct  |  N={len(y)}  base rate correct={y.mean():.3f}")
print(f"(prompt-only ceiling for easy-vs-hard was AUC 0.59)")
print("=" * 60)
print("CONFOUND CONTROLS:")
auc(temp, "temperature only")
auc(rlen, "response length only")
auc(np.concatenate([temp, rlen], 1), "temp + length")
print("BASELINE:")
auc(p_hidden, "PROMPT-ONLY hidden (structural ~0.5)")
print("REASONING FEATURES:")
auc(h_mean, "reasoning hidden_mean")
auc(h_last, "reasoning hidden_last")
auc(np.concatenate([h_last, temp, rlen], 1), "hidden_last + temp + length")
print("=" * 60)
