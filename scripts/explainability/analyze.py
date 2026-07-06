"""Per-model analysis of the clean-data correctness signal (runs on the Mac).

Inputs (from collect.py):
  data/gsm8k_en_clean.{model}.train.pt   fit split (no layer sweep)
  data/gsm8k_en_clean.{model}.test.pt    eval split (with --layers)

If the train .pt is missing (e.g. phase-1 sanity runs), everything falls back
to GroupKFold CV within the test split, flagged in the output as mode="cv".

Per temperature band:
  A  END SIGNAL      arrow + probe fit on train -> AUC on test, with cluster
                     bootstrap 95% CIs (resampling test prompts); CV-refit
                     ceilings within the test split.
  B  POSITIONAL      native arrow per frac (incl. 90/95%) and the end-arrow
                     transferred to each frac; the PRE-ANSWER tap ("knows
                     before it commits?") on rows with a #### marker.
  C  BASELINES       response length, mean/min token logprob, first-quarter
                     logprob (early confidence); hidden vs hidden+baselines.
  D  CALIBRATION     probe fit on train -> test ECE + Brier.

Across bands:
  E  TEMP-SHIFT      arrow/probe fit on train band i -> test band j (the
                     robustness-under-distribution-shift matrix).
  F  GEOMETRY        cosine(arrow_i, arrow_j) across bands; cos(pre, end).
  G  LAYER HEATMAP   arrow AUC per (layer, position) via CV on the test split.

Writes paper/analysis.{model}.json.

Usage:
    python analyze.py --model qwen
    python analyze.py --model ministral --bootstrap 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data"

MIN_PER_CLASS = 30
N_SPLITS = 5
ECE_BINS = 10
RNG = np.random.default_rng(0)


# ============================================================
# Small estimators
# ============================================================

def arrow(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def make_probe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))


def cv_auc_arrow(X, y, g) -> float:
    aucs = []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        aucs.append(roc_auc_score(y[te], X[te] @ arrow(X[tr], y[tr])))
    return float(np.mean(aucs))


def cv_auc_probe(X, y, g) -> float:
    clf = make_probe()
    aucs = []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, g):
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def cluster_bootstrap_ci(y: np.ndarray, s: np.ndarray, g: np.ndarray, B: int) -> list[float]:
    """95% CI for AUC(y, s), resampling PROMPTS (clusters) with replacement."""
    uniq = np.unique(g)
    rows_of = {u: np.where(g == u)[0] for u in uniq}
    aucs = []
    for _ in range(B):
        take = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_of[u] for u in take])
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [float(lo), float(hi)]


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def fmt_ci(entry: dict, key: str) -> str:
    ci = entry.get(key)
    return f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""


# ============================================================
# Data access
# ============================================================

class Split:
    """One split's tensors, truncation-filtered, with typed accessors."""

    def __init__(self, path: Path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        keep = ~d["truncated"].numpy()
        self.meta = d["meta"]
        self.fracs = [float(f) for f in d["fracs"]]
        self.y = d["was_correct"].numpy().astype(int)[keep]
        self.t = d["temperature"].numpy()[keep]
        self.g = d["index"].numpy()[keep]
        self.rlen = d["resp_len"].numpy().astype(float)[keep]
        self.marker = d["has_marker"].numpy()[keep]
        self.pre_frac = d["pre_frac"].numpy()[keep]
        self.lp = d["lp_stats"].numpy()[keep]          # [:, (mean, min, sum)]
        self.lp_q = d["lp_quarters"].numpy()[keep]
        self._hf = d["hidden_frac"][keep]              # fp16, sliced lazily
        self._hp = d["hidden_pre"][keep]
        self._hl = d.get("hidden_layers")
        if self._hl is not None:
            self._hl = self._hl[keep]
            self.layer_taps = list(d["meta"]["layer_taps"])
            self.layer_positions = list(d["meta"]["layer_positions"])
        self.n_dropped = int((~keep).sum())

    def H(self, fi: int) -> np.ndarray:
        return self._hf[:, fi, :].numpy().astype(np.float32)

    def Hpre(self) -> np.ndarray:
        return self._hp.numpy().astype(np.float32)

    def Hlayer(self, li: int, pi: int) -> np.ndarray:
        return self._hl[:, li, pi, :].numpy().astype(np.float32)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ministral", "qwen", "llama"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    test_path = DATA / f"gsm8k_en_clean.{args.model}.test.pt"
    train_path = DATA / f"gsm8k_en_clean.{args.model}.train.pt"
    out_path = ROOT / "paper" / f"analysis.{args.model}.json"

    TE = Split(test_path)
    TR = Split(train_path) if train_path.exists() else None
    mode = "split" if TR is not None else "cv"
    print(f"Model: {args.model}  mode: {mode}"
          + (f"  (train rows: {len(TR.y)}, dropped {TR.n_dropped} truncated)" if TR else ""))
    print(f"Test rows: {len(TE.y)} (dropped {TE.n_dropped} truncated)  "
          f"fracs: {TE.fracs}")

    fracs = TE.fracs
    F_END = fracs.index(1.0)
    bands = sorted(set(TE.t.tolist()))
    B = args.bootstrap

    results: dict = {"model": args.model, "mode": mode, "fracs": fracs,
                     "meta": {"test": dict(TE.meta), **({"train": dict(TR.meta)} if TR else {})},
                     "bands": {}, "temp_shift": {}, "geometry": {}, "layers": {}}
    for k in ("test", "train"):
        results["meta"].get(k, {}).pop("layer_taps", None)

    def fit_eval(X_fit, y_fit, X_te, y_te, g_te, ci=False):
        """arrow + probe scores: fit on (X_fit,y_fit), evaluate on the test rows.
        In cv mode the caller passes the SAME band twice and we CV instead."""
        out = {}
        if mode == "split":
            w = arrow(X_fit, y_fit)
            s_arrow = X_te @ w
            clf = make_probe().fit(X_fit, y_fit)
            s_probe = clf.predict_proba(X_te)[:, 1]
            out["arrow_auc"] = float(roc_auc_score(y_te, s_arrow))
            out["probe_auc"] = float(roc_auc_score(y_te, s_probe))
            if ci:
                out["arrow_ci"] = cluster_bootstrap_ci(y_te, s_arrow, g_te, B)
                out["probe_ci"] = cluster_bootstrap_ci(y_te, s_probe, g_te, B)
        else:
            out["arrow_auc"] = cv_auc_arrow(X_te, y_te, g_te)
            out["probe_auc"] = cv_auc_probe(X_te, y_te, g_te)
        return out

    # ---------- per-band: end signal, positions, baselines, calibration ----------
    band_arrows: dict[float, np.ndarray] = {}   # end arrows for geometry/shift
    for tv in bands:
        mt = TE.t == tv
        y_te, g_te = TE.y[mt], TE.g[mt]
        if min(y_te.sum(), (1 - y_te).sum()) < MIN_PER_CLASS:
            print(f"\ntau={tv:.1f}: skipped (test correct={int(y_te.sum())}, "
                  f"incorrect={int((1-y_te).sum())})")
            continue
        if TR is not None:
            mf = TR.t == tv
            y_fit, g_fit = TR.y[mf], TR.g[mf]
            if min(y_fit.sum(), (1 - y_fit).sum()) < MIN_PER_CLASS:
                print(f"\ntau={tv:.1f}: skipped (train too thin)")
                continue
        else:
            mf, y_fit, g_fit = mt, y_te, g_te

        SRC = TR if TR is not None else TE
        band: dict = {"n_test": int(mt.sum()), "test_base_rate": float(y_te.mean()),
                      "n_fit": int(mf.sum()), "fit_base_rate": float(y_fit.mean())}

        # A — end-of-reasoning signal
        Xf_end, Xt_end = SRC.H(F_END)[mf], TE.H(F_END)[mt]
        band["end"] = fit_eval(Xf_end, y_fit, Xt_end, y_te, g_te, ci=True)
        band["end"]["refit_arrow_cv"] = cv_auc_arrow(Xt_end, y_te, g_te)
        band["end"]["refit_probe_cv"] = cv_auc_probe(Xt_end, y_te, g_te)
        w_end = arrow(Xf_end, y_fit)
        band_arrows[tv] = w_end

        # B — positional curve + pre-answer tap
        band["positions"] = {}
        for fi, fv in enumerate(fracs):
            Xt = TE.H(fi)[mt]
            entry = {"end_arrow_auc": float(roc_auc_score(y_te, Xt @ w_end))}
            entry["native_arrow_auc"] = (
                fit_eval(SRC.H(fi)[mf], y_fit, Xt, y_te, g_te)["arrow_auc"])
            band["positions"][f"{fv:.2f}"] = entry
        # pre-answer: only rows where the tap is genuinely before a #### marker
        mk_te = mt & TE.marker
        mk_fit = mf & SRC.marker
        y_mk, g_mk = TE.y[mk_te], TE.g[mk_te]
        if min(y_mk.sum(), (1 - y_mk).sum()) >= MIN_PER_CLASS:
            Xt_pre = TE.Hpre()[mk_te]
            pre = fit_eval(SRC.Hpre()[mk_fit], SRC.y[mk_fit], Xt_pre, y_mk, g_mk, ci=True)
            pre["end_arrow_auc"] = float(roc_auc_score(y_mk, Xt_pre @ w_end))
            pre["n"] = int(mk_te.sum())
            pre["mean_pre_frac"] = float(TE.pre_frac[mk_te].mean())
            band["pre_answer"] = pre

        # C — baselines (orientation fixed a priori: shorter/likelier -> correct)
        band["baselines"] = {
            "neg_len_auc": float(roc_auc_score(y_te, -TE.rlen[mt])),
            "neg_len_ci": cluster_bootstrap_ci(y_te, -TE.rlen[mt], g_te, B),
            "lp_mean_auc": float(roc_auc_score(y_te, TE.lp[mt, 0])),
            "lp_mean_ci": cluster_bootstrap_ci(y_te, TE.lp[mt, 0], g_te, B),
            "lp_min_auc": float(roc_auc_score(y_te, TE.lp[mt, 1])),
            "lp_q1_auc": float(roc_auc_score(y_te, TE.lp_q[mt, 0])),
        }
        base_fit = np.column_stack([SRC.rlen[mf], SRC.lp[mf, 0], SRC.lp[mf, 1]])
        base_te = np.column_stack([TE.rlen[mt], TE.lp[mt, 0], TE.lp[mt, 1]])
        band["baselines"]["hidden_plus_baselines_auc"] = fit_eval(
            np.hstack([Xf_end, base_fit]), y_fit,
            np.hstack([Xt_end, base_te]), y_te, g_te)["probe_auc"]

        # D — calibration of the probe on the test band
        if mode == "split":
            clf = make_probe().fit(Xf_end, y_fit)
            p = clf.predict_proba(Xt_end)[:, 1]
        else:
            clf = make_probe()
            p = np.zeros(len(y_te))
            for tr_i, te_i in GroupKFold(N_SPLITS).split(Xt_end, y_te, g_te):
                clf.fit(Xt_end[tr_i], y_te[tr_i])
                p[te_i] = clf.predict_proba(Xt_end[te_i])[:, 1]
        band["calibration"] = {"ece": ece(y_te, p), "brier": float(brier_score_loss(y_te, p))}

        results["bands"][f"{tv:.1f}"] = band

        e = band["end"]
        print(f"\ntau={tv:.1f}  n_test={band['n_test']}  base_rate={band['test_base_rate']:.2f}")
        print(f"  A end   : arrow={e['arrow_auc']:.3f} {fmt_ci(e, 'arrow_ci')}  "
              f"probe={e['probe_auc']:.3f} {fmt_ci(e, 'probe_ci')}  "
              f"(refit-CV ceiling: arrow={e['refit_arrow_cv']:.3f} probe={e['refit_probe_cv']:.3f})")
        pos = "  ".join(f"{int(fv*100)}%={band['positions'][f'{fv:.2f}']['native_arrow_auc']:.3f}"
                        for fv in fracs)
        print(f"  B native: {pos}")
        if "pre_answer" in band:
            pa = band["pre_answer"]
            print(f"  B pre   : arrow={pa['arrow_auc']:.3f} {fmt_ci(pa, 'arrow_ci')}  "
                  f"(n={pa['n']}, mean pre_frac={pa['mean_pre_frac']:.2f})")
        bl = band["baselines"]
        print(f"  C base  : -len={bl['neg_len_auc']:.3f}  lp_mean={bl['lp_mean_auc']:.3f}  "
              f"lp_min={bl['lp_min_auc']:.3f}  lp_q1={bl['lp_q1_auc']:.3f}  "
              f"hidden+base={bl['hidden_plus_baselines_auc']:.3f}")
        print(f"  D calib : ECE={band['calibration']['ece']:.3f}  "
              f"Brier={band['calibration']['brier']:.3f}")

    # ---------- E: temperature-shift matrix (fit band i -> eval band j) ----------
    usable = sorted(band_arrows)
    for ti in usable:
        SRC = TR if TR is not None else TE
        mf = (SRC.t == ti)
        row_a, row_p = {}, {}
        clf = make_probe().fit(SRC.H(F_END)[mf], SRC.y[mf])
        for tj in usable:
            mt = TE.t == tj
            y_te = TE.y[mt]
            Xt = TE.H(F_END)[mt]
            row_a[f"{tj:.1f}"] = float(roc_auc_score(y_te, Xt @ band_arrows[ti]))
            row_p[f"{tj:.1f}"] = float(roc_auc_score(y_te, clf.predict_proba(Xt)[:, 1]))
        results["temp_shift"][f"{ti:.1f}"] = {"arrow": row_a, "probe": row_p}
    if usable:
        print("\nE temp-shift (fit tau -> eval tau), arrow | probe:")
        for ti in usable:
            r = results["temp_shift"][f"{ti:.1f}"]
            print(f"  fit {ti:.1f}: " + "  ".join(
                f"->{tj:.1f} {r['arrow'][f'{tj:.1f}']:.3f}|{r['probe'][f'{tj:.1f}']:.3f}"
                for tj in usable))

    # ---------- F: arrow geometry across bands + pre/end ----------
    if len(usable) >= 2:
        cosm = {f"{a:.1f}": {f"{b:.1f}": float(unit(band_arrows[a]) @ unit(band_arrows[b]))
                             for b in usable} for a in usable}
        results["geometry"]["cos_band_arrows"] = cosm
        print("F cos(arrow_i, arrow_j): " + "  ".join(
            f"{a:.1f}~{b:.1f}={cosm[f'{a:.1f}'][f'{b:.1f}']:.2f}"
            for i, a in enumerate(usable) for b in usable[i+1:]))
    SRC = TR if TR is not None else TE
    mk = SRC.marker
    if mk.sum() > 2 * MIN_PER_CLASS:
        w_pre = arrow(SRC.Hpre()[mk], SRC.y[mk])
        w_all_end = arrow(SRC.H(F_END), SRC.y)
        results["geometry"]["cos_pre_end"] = float(unit(w_pre) @ unit(w_all_end))
        print(f"F cos(pre-answer arrow, end arrow): {results['geometry']['cos_pre_end']:.2f}")

    # ---------- G: layer x position heatmap (CV within test) ----------
    if TE._hl is not None:
        print("\nG layer heatmap (arrow CV AUC, test split):")
        for tv in usable:
            mt = TE.t == tv
            y_te, g_te = TE.y[mt], TE.g[mt]
            grid = {}
            for li, layer in enumerate(TE.layer_taps):
                grid[str(layer)] = {p: cv_auc_arrow(TE.Hlayer(li, pi)[mt], y_te, g_te)
                                    for pi, p in enumerate(TE.layer_positions)}
            results["layers"][f"{tv:.1f}"] = {"taps": TE.layer_taps,
                                              "positions": TE.layer_positions, "auc": grid}
            for layer in TE.layer_taps:
                row = grid[str(layer)]
                print(f"  tau={tv:.1f} L{layer:>2}: " +
                      "  ".join(f"{p}={row[p]:.3f}" for p in TE.layer_positions))

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
