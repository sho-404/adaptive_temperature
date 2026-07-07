"""Direction geometry, with error bars.

(1) PRE-ANSWER TEMP-TRANSFER: the end arrow is temperature-invariant; is the
    pre-answer arrow too? Fit the pre-answer arrow on train band tau_i, eval
    on test band tau_j (marker rows only) -> transfer matrix, like the end tap.
(2) COSINE CIs: bootstrap (over train prompts) 95% CIs for
      cos(arrow_tau_i, arrow_tau_j)   at the end tap and at the pre tap
      cos(arrow_pre, arrow_end)       pooled over bands
    The 0.93-0.99 "same axis across temperature" and 0.19-0.32 "different
    axis before vs after the answer" claims get error bars.

Runs locally off the clean train/test .pt. Writes paper/geometry_ci.json.
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
OUT = ROOT / "paper" / "geometry_ci.json"

MODELS = ["ministral", "qwen", "llama"]
BANDS = [0.0, 0.6, 1.0]
B = 300
RNG = np.random.default_rng(0)


def load(model: str, split: str):
    d = torch.load(DATA / f"gsm8k_en_clean.{model}.{split}.pt",
                   map_location="cpu", weights_only=False)
    keep = (~d["truncated"].numpy())
    fracs = [float(f) for f in d["fracs"]]
    return {
        "end": d["hidden_frac"][:, fracs.index(1.0), :].numpy().astype(np.float32)[keep],
        "pre": d["hidden_pre"].numpy().astype(np.float32)[keep],
        "marker": d["has_marker"].numpy()[keep],
        "y": d["was_correct"].numpy().astype(int)[keep],
        "t": d["temperature"].numpy()[keep],
        "g": d["index"].numpy()[keep],
    }


def arrow(X, y):
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def unit(v):
    return v / np.linalg.norm(v)


def band_arrows(S, tap: str, mask_extra=None):
    out = {}
    for tv in BANDS:
        m = S["t"] == tv
        if mask_extra is not None:
            m &= mask_extra
        out[tv] = arrow(S[tap][m], S["y"][m])
    return out


results = {}
for model in MODELS:
    TR, TE = load(model, "train"), load(model, "test")
    res: dict = {}

    # (1) pre-answer temperature-transfer matrix
    pre_arrows = band_arrows(TR, "pre", TR["marker"])
    mat = {}
    for ti, w in pre_arrows.items():
        row = {}
        for tj in BANDS:
            m = (TE["t"] == tj) & TE["marker"]
            row[f"{tj:.1f}"] = float(roc_auc_score(TE["y"][m], TE["pre"][m] @ w))
        mat[f"{ti:.1f}"] = row
    res["pre_transfer"] = mat

    # (2) bootstrap cosines over train prompts
    rows_of = defaultdict(list)
    for i, gg in enumerate(TR["g"]):
        rows_of[gg].append(i)
    prompts = np.array(list(rows_of.keys()))
    pairs = [(0.0, 0.6), (0.0, 1.0), (0.6, 1.0)]
    cos_samples = {("end",) + p: [] for p in pairs}
    cos_samples.update({("pre",) + p: [] for p in pairs})
    cos_pre_end = []
    for _ in range(B):
        take = RNG.choice(prompts, size=len(prompts), replace=True)
        idx = np.concatenate([rows_of[p] for p in take])
        S = {k: TR[k][idx] for k in ("end", "pre", "marker", "y", "t")}
        try:
            a_end = band_arrows(S, "end")
            a_pre = band_arrows(S, "pre", S["marker"])
        except Exception:
            continue  # a resample lost a class in some band; skip it
        for p in pairs:
            cos_samples[("end",) + p].append(float(unit(a_end[p[0]]) @ unit(a_end[p[1]])))
            cos_samples[("pre",) + p].append(float(unit(a_pre[p[0]]) @ unit(a_pre[p[1]])))
        mk = S["marker"]
        cos_pre_end.append(float(unit(arrow(S["pre"][mk], S["y"][mk]))
                                 @ unit(arrow(S["end"], S["y"]))))

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return [float(np.mean(v)), float(lo), float(hi)]

    res["cos_end_bands"] = {f"{a:.1f}~{b:.1f}": ci(cos_samples[("end", a, b)]) for a, b in pairs}
    res["cos_pre_bands"] = {f"{a:.1f}~{b:.1f}": ci(cos_samples[("pre", a, b)]) for a, b in pairs}
    res["cos_pre_vs_end"] = ci(cos_pre_end)
    results[model] = res

    print(f"\n{model}:")
    print("  pre-answer transfer (fit tau -> eval tau):")
    for ti in BANDS:
        r = mat[f"{ti:.1f}"]
        print(f"    fit {ti:.1f}: " + "  ".join(f"->{tj:.1f} {r[f'{tj:.1f}']:.3f}" for tj in BANDS))
    fmt = lambda d: "  ".join(f"{k}={v[0]:.2f}[{v[1]:.2f},{v[2]:.2f}]" for k, v in d.items())
    print(f"  cos end-arrows across bands: {fmt(res['cos_end_bands'])}")
    print(f"  cos pre-arrows across bands: {fmt(res['cos_pre_bands'])}")
    c = res["cos_pre_vs_end"]
    print(f"  cos(pre, end) pooled: {c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]")

OUT.write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT.relative_to(ROOT)}")
