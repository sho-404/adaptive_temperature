"""Generate all paper figures from paper/*.json and data/*.pt.

Outputs vector PDFs (+ PNG proofs) into paper_overleaf/figures/.
Run from repo root: venv/bin/python scripts/figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_overleaf" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["ministral", "qwen", "llama"]
LABEL = {"ministral": "Ministral-3-3B", "qwen": "Qwen2.5-3B", "llama": "Llama-3.2-3B"}
BANDS = ["0.0", "0.6", "1.0"]

# palette (validated): status pair for correct/incorrect, ordinal blues for
# temperature, categorical slots for models/series (relief rule -> direct labels)
GOOD, CRIT = "#0ca30c", "#d03b3b"
TEMP_C = {"0.0": "#86b6ef", "0.6": "#2a78d6", "1.0": "#104281"}
CAT = ["#2a78d6", "#1baf7a", "#eda100"]
INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seqblue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

COL_W, PAGE_W = 3.45, 7.16


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def load_json(name):
    return json.load(open(ROOT / "paper" / name))


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- pipeline
def fig_pipeline():
    """Five-stage process diagram (drawn here instead of TikZ: the IEEE
    Access class's spot-color setup breaks if pgf is loaded)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    stages = [
        ("Generate", "GSM8K train+test;\nper prompt: 1 greedy\n+ 2@$\\tau$0.6 + 2@$\\tau$1.0\n$\\approx$44k chains/model"),
        ("Grade", "parse final '####';\nexact match $\\Rightarrow$\ncorrect / incorrect;\ndrop truncated"),
        ("Teacher-force\n& tap", "re-forward own chain;\nresidual stream at\n25–100%, pre-answer\ntoken, 8 layers; log-probs"),
        ("Fit on train", "arrow $\\mathbf{w}=\\mu_+ - \\mu_-$\nand logistic probe,\nper temperature band"),
        ("Evaluate\non test", "AUC + bootstrap CIs;\n3$\\times$3 temp transfer;\nwithin-prompt pairs;\ncosines; selective pred."),
    ]
    notes = [
        (1.5, "every chain is regraded from its own text;\nlabels are never taken from the sampler"),
        (3.5, "fit and evaluation splits share no prompts;\nrepeated for Ministral-3-3B, Qwen2.5-3B, Llama-3.2-3B"),
    ]
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.15))
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-0.62, 0.62)
    ax.axis("off")
    ax.grid(False)
    for i, (title, body) in enumerate(stages):
        last = i == len(stages) - 1
        ax.add_patch(FancyBboxPatch(
            (i - 0.44, -0.40), 0.88, 0.80,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            fc="white" if last else "#eaf2fc",
            ec="#104281" if last else "#2a78d6", lw=1.2, zorder=2))
        ax.text(i, 0.28 if "\n" not in title else 0.245, title,
                ha="center", va="center",
                fontsize=6.9, fontweight="bold", color=INK, zorder=3,
                linespacing=1.15)
        ax.text(i, -0.12, body, ha="center", va="center",
                fontsize=5.5, color=INK, zorder=3, linespacing=1.4)
        ax.text(i - 0.44, 0.46, f" {i + 1} ", ha="left", va="center",
                fontsize=6.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="square,pad=0.18", fc="#2a78d6", ec="none"),
                zorder=4)
        if not last:
            ax.add_patch(FancyArrowPatch(
                (i + 0.47, 0.0), (i + 0.53, 0.0),
                arrowstyle="-|>", mutation_scale=11,
                color=MUTED, lw=1.2, zorder=2))
    for x, txt in notes:
        ax.text(x, -0.55, txt, ha="center", va="center",
                fontsize=5.8, color=MUTED, style="italic", linespacing=1.3)
    save(fig, "fig_pipeline")


# ---------------------------------------------------------------- geometry
def fig_geometry():
    """Ministral end-of-reasoning states projected on the FROZEN tau=0 arrow
    (x) and the top orthogonal PC (y), one panel per sampling temperature."""
    tr = torch.load(ROOT / "data/gsm8k_en_clean.ministral.train.pt",
                    map_location="cpu", weights_only=False)
    te = torch.load(ROOT / "data/gsm8k_en_clean.ministral.test.pt",
                    map_location="cpu", weights_only=False)

    def end_h(d):
        return d["hidden_frac"][:, -1, :].float().numpy()

    def ok(d):
        return (~d["truncated"]).numpy()      # match analyze.py's filter

    Xtr, ytr = end_h(tr), tr["was_correct"].numpy()
    ttr, mtr = tr["temperature"].numpy(), ok(tr)
    sel = mtr & (ttr == 0.0)
    arrow = Xtr[sel & ytr].mean(0) - Xtr[sel & ~ytr].mean(0)
    arrow /= np.linalg.norm(arrow)
    R = Xtr[sel] - Xtr[sel].mean(0)
    R = R - np.outer(R @ arrow, arrow)          # remove arrow component
    _, _, Vt = np.linalg.svd(R[:4000], full_matrices=False)
    pc = Vt[0]

    Xte, yte = end_h(te), te["was_correct"].numpy()
    tte, mte = te["temperature"].numpy(), ok(te)
    mu = Xtr[mtr].mean(0)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.5), sharex=True, sharey=True)
    for ax, b in zip(axes, BANDS):
        m = mte & np.isclose(tte, float(b))
        X, y = Xte[m] - mu, yte[m]
        px, py = X @ arrow, X @ pc
        auc = roc_auc_score(y, px)
        keep = rng.permutation(len(y))[:1200]
        for cls, col, mk, lab, al, z in [
                (True, GOOD, "o", "correct", 0.30, 2),
                (False, CRIT, "x", "incorrect", 0.75, 3)]:
            i = keep[y[keep] == cls]
            ax.scatter(px[i], py[i], s=6 if cls else 10, marker=mk, c=col,
                       alpha=al, linewidths=0.8, label=lab, zorder=z)
        thr = 0.5 * (px[y].mean() + px[~y].mean())
        ax.axvline(thr, color=INK, lw=0.8, ls="--")
        ax.set_title(f"$\\tau={b}$   AUC {auc:.2f}")
        ax.set_xlabel("projection on frozen $\\tau{=}0$ arrow")
        despine(ax)
    axes[0].set_ylabel("top orthogonal PC")
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.2)
    fig.suptitle("Ministral-3-3B: one frozen direction separates all temperature bands",
                 fontsize=9, y=1.02)
    save(fig, "fig_geometry")


# ---------------------------------------------------------------- positional
def fig_positional():
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.3), sharey=True)
    xs = [0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
    for ax, m in zip(axes, MODELS):
        d = load_json(f"analysis.{m}.json")
        for b in BANDS:
            v = d["bands"][b]
            ys = [v["positions"][f"{x:.2f}"]["native_arrow_auc"] for x in xs]
            ax.plot(xs, ys, "-o", ms=3, lw=1.4, color=TEMP_C[b],
                    label=f"$\\tau={b}$")
            ax.plot([0.98], [v["pre_answer"]["arrow_auc"]], marker="D", ms=4,
                    color=TEMP_C[b], mec="white", mew=0.5, zorder=5)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_title(LABEL[m])
        ax.set_xticks([0.25, 0.50, 0.75, 1.00])
        ax.set_xticklabels(["25%", "50%", "75%", "end"])
        ax.set_xticks([0.90, 0.95], minor=True)
        ax.set_xlabel("position in reasoning chain")
        ax.set_ylim(0.45, 0.88)
        despine(ax)
    axes[0].set_ylabel("AUC (arrow, held-out)")
    axes[0].annotate("pre-answer tap", xy=(0.975, 0.775), xytext=(0.62, 0.83),
                     fontsize=7, color=INK,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
    axes[2].legend(frameon=False, loc="upper left")
    save(fig, "fig_positional")


# ---------------------------------------------------------------- temp shift
def fig_tempshift():
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.35))
    for ax, m in zip(axes, MODELS):
        d = load_json(f"analysis.{m}.json")["temp_shift"]
        M = np.array([[d[a]["arrow"][b] for b in BANDS] for a in BANDS])
        im = ax.imshow(M, cmap=SEQ, vmin=0.55, vmax=0.90)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if M[i, j] > 0.75 else INK)
        ax.set_xticks(range(3), [f"{b}" for b in BANDS])
        ax.set_yticks(range(3), [f"{b}" for b in BANDS])
        ax.set_xlabel("evaluated on $\\tau$")
        if m == "ministral":
            ax.set_ylabel("arrow fit on $\\tau$")
        ax.set_title(LABEL[m])
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="AUC", pad=0.02)
    save(fig, "fig_tempshift")


# ---------------------------------------------------------------- within-prompt
def fig_within_prompt():
    d = load_json("within_prompt.json")
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    w = 0.26
    xs = np.arange(3)
    for k, (key, lab, col) in enumerate([
            ("pooled_auc", "pooled AUC", CAT[0]),
            ("within_mixed", "within-prompt acc. (mixed $\\tau$)", CAT[1]),
            ("within_same_band", "within-prompt acc. (same $\\tau$)", CAT[2])]):
        vals, errs = [], []
        for m in MODELS:
            if key == "pooled_auc":
                vals.append(d[m][key]); errs.append([0, 0])
            else:
                v = d[m][key]
                vals.append(v["acc"])
                errs.append([v["acc"] - v["ci"][0], v["ci"][1] - v["acc"]])
        err = np.array(errs).T
        ax.bar(xs + (k - 1) * w, vals, w, color=col, label=lab,
               yerr=None if key == "pooled_auc" else err,
               error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
        for x, v in zip(xs + (k - 1) * w, vals):
            ax.text(x, 0.52, f"{v:.2f}", ha="center", fontsize=6.5,
                    color="white", rotation=90, va="bottom")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(xs, [LABEL[m] for m in MODELS])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUC / pairwise accuracy")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    despine(ax)
    save(fig, "fig_within_prompt")


# ---------------------------------------------------------------- risk-coverage
def fig_riskcov():
    d = load_json("verifier.json")
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for m, col in zip(MODELS, CAT):
        rc = d[m]["risk_coverage_best_of_5"]
        cov = [int(k.rstrip("%")) for k in rc]
        acc = list(rc.values())
        order = np.argsort(cov)
        cov, acc = np.array(cov)[order], np.array(acc)[order]
        ax.plot(cov, acc, "-o", ms=3.5, lw=1.4, color=col)
        ax.annotate(LABEL[m], (cov[0], acc[0]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=INK)
    ax.set_xlabel("coverage (% of problems answered)")
    ax.set_ylabel("accuracy on answered subset")
    ax.set_xlim(105, 45)          # read left-to-right as "abstain more"
    despine(ax)
    save(fig, "fig_riskcov")


# ---------------------------------------------------------------- layers
def fig_layers():
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.3), sharey=True)
    for ax, m in zip(axes, MODELS):
        d = load_json(f"analysis.{m}.json")
        L = d["layers"]["1.0"]["auc"]
        n_layers = d["meta"]["test"]["n_layers"] if m != "qwen" else max(map(int, L))
        taps = sorted(map(int, L))
        for pos, col, lab in [("mid", CAT[0], "mid-chain (50%)"),
                              ("pre", CAT[1], "pre-answer"),
                              ("end", CAT[2], "end of chain")]:
            ys = [L[str(t)][pos] for t in taps]
            ax.plot(np.array(taps) / taps[-1], ys, "-o", ms=3, lw=1.4,
                    color=col, label=lab)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_title(LABEL[m])
        ax.set_xlabel("relative layer depth")
        ax.set_ylim(0.45, 0.88)
        despine(ax)
    axes[0].set_ylabel("AUC (arrow, $\\tau{=}1$)")
    axes[2].legend(frameon=False, loc="lower right", fontsize=6.5)
    save(fig, "fig_layers")


# ---------------------------------------------------------------- robustness
def fig_robustness():
    d = load_json("clean_validation.json")["bands"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    w = 0.26
    xs = np.arange(3)
    series = [("frozen arrow", [d[b]["arrow_auc"] for b in BANDS], CAT[0]),
              ("frozen probe", [d[b]["probe_auc"] for b in BANDS], CAT[1]),
              ("refit ceiling", [max(d[b]["refit"]["arrow_auc"],
                                     d[b]["refit"]["probe_auc"]) for b in BANDS], CAT[2])]
    for k, (lab, vals, col) in enumerate(series):
        ax.bar(xs + (k - 1) * w, vals, w, color=col, label=lab)
        for x, v in zip(xs + (k - 1) * w, vals):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(xs, [f"$\\tau={b}$" for b in BANDS])
    ax.set_ylim(0.40, 0.85)
    ax.set_ylabel("AUC on clean generations")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    despine(ax)
    save(fig, "fig_robustness")


# ---------------------------------------------------------------- axes cosines
def fig_axes_cos():
    g = load_json("geometry_ci.json")
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    w = 0.26
    xs = np.arange(3)
    pairs = [("0.0~0.6", "$\\tau$ 0–0.6"), ("0.0~1.0", "$\\tau$ 0–1"),
             ("0.6~1.0", "$\\tau$ 0.6–1")]
    shades = ["#86b6ef", "#2a78d6", "#104281"]
    for k, ((key, lab), col) in enumerate(zip(pairs, shades)):
        vals = [g[m]["cos_end_bands"][key][0] for m in MODELS]
        los = [g[m]["cos_end_bands"][key][1] for m in MODELS]
        his = [g[m]["cos_end_bands"][key][2] for m in MODELS]
        err = [np.array(vals) - los, np.array(his) - np.array(vals)]
        ax.bar(xs + (k - 1.5) * w * 0.85, vals, w * 0.8, color=col, label=lab,
               yerr=err, error_kw=dict(ecolor=INK, lw=0.7, capsize=1.5))
    vals = [g[m]["cos_pre_vs_end"][0] for m in MODELS]
    los = [g[m]["cos_pre_vs_end"][1] for m in MODELS]
    his = [g[m]["cos_pre_vs_end"][2] for m in MODELS]
    err = [np.array(vals) - los, np.array(his) - np.array(vals)]
    ax.bar(xs + 1.5 * w * 0.85, vals, w * 0.8, color=CRIT,
           label="pre vs. end axis", yerr=err,
           error_kw=dict(ecolor=INK, lw=0.7, capsize=1.5))
    ax.set_xticks(xs, [LABEL[m] for m in MODELS])
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handletextpad=0.4)
    despine(ax)
    save(fig, "fig_axes_cos")


if __name__ == "__main__":
    fig_pipeline()
    fig_geometry()
    fig_positional()
    fig_tempshift()
    fig_within_prompt()
    fig_riskcov()
    fig_layers()
    fig_robustness()
    fig_axes_cos()


# ================================================================ v2 figures
def _ministral_directions():
    """The four unit directions (end arrows per band + pre-answer arrow),
    fitted on the train split exactly as analyze.py fits them."""
    tr = torch.load(ROOT / "data/gsm8k_en_clean.ministral.train.pt",
                    map_location="cpu", weights_only=False)
    keep = (~tr["truncated"]).numpy()
    mk = tr["has_marker"].numpy()[keep]
    y = tr["was_correct"].numpy()[keep]
    t = tr["temperature"].numpy()[keep]
    Xe = tr["hidden_frac"][:, -1, :].float().numpy()[keep]
    Xp = tr["hidden_pre"].float().numpy()[keep]

    def unit(v):
        return v / np.linalg.norm(v)

    ends = {b: unit(Xe[(m := np.isclose(t, b)) & y].mean(0)
                    - Xe[m & ~y].mean(0)) for b in (0.0, 0.6, 1.0)}
    pre = unit(Xp[mk & y].mean(0) - Xp[mk & ~y].mean(0))
    return ends, pre, (Xe, y, t)


def fig_axes3d():
    """Hero geometry figure: the four directions drawn as vectors in the
    3-D subspace they (almost exactly) span; same scene from three angles.
    Angles between the drawn vectors are the TRUE high-dim cosines."""
    ends, pre, _ = _ministral_directions()
    V = np.stack([ends[0.0], ends[0.6], ends[1.0], pre])
    # Gram-Schmidt basis: e1 = bundle mean (blues lie along +x), e2 = the
    # part of the pre-arrow orthogonal to it (red opens into +y), e3 = rest.
    e1 = V[:3].mean(0); e1 /= np.linalg.norm(e1)
    e2 = pre - (pre @ e1) * e1; e2 /= np.linalg.norm(e2)
    r = V[2] - (V[2] @ e1) * e1 - (V[2] @ e2) * e2
    e3 = r / np.linalg.norm(r)
    P = V @ np.stack([e1, e2, e3]).T      # exact up to <0.2% norm loss
    labels = ["end arrow, $\\tau{=}0$", "end arrow, $\\tau{=}0.6$",
              "end arrow, $\\tau{=}1$", "pre-answer arrow"]
    colors = [TEMP_C["0.0"], TEMP_C["0.6"], TEMP_C["1.0"], CRIT]

    fig = plt.figure(figsize=(PAGE_W, 2.75))
    for k, azim in enumerate((15, 50, 85)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for v, c, lab in zip(P, colors, labels):
            ax.quiver(0, 0, 0, *v, color=c, lw=2.2, arrow_length_ratio=0.09,
                      label=lab if k == 0 else None)
        ax.scatter([0], [0], [0], color=INK, s=8)
        # equal ranges on all three axes so angles are not distorted
        ax.set_xlim(-0.25, 1.05); ax.set_ylim(-0.25, 1.05); ax.set_zlim(-0.65, 0.65)
        ax.view_init(elev=18, azim=azim)
        ax.set_proj_type("ortho")         # faithful angles, no perspective
        ax.set_title(f"view {k+1} (rotated {35*k}$^\\circ$)", fontsize=8)
        ax.set_box_aspect((1, 1, 1))
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_ticklabels([])
            pane.pane.set_alpha(0.05)
            pane.line.set_color(BASE)
        ax.grid(False)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("The three temperature arrows are one axis; the pre-answer "
                 "arrow is another (cosines 0.96--0.99 vs. 0.19)",
                 fontsize=8.5, y=0.99)
    save(fig, "fig_axes3d")


def fig_ridges():
    """Score densities under the ONE frozen tau=0 arrow, per band: the same
    ruler separates correct from incorrect at every temperature."""
    ends, _, (Xe_tr, y_tr, t_tr) = _ministral_directions()
    w = ends[0.0]
    te = torch.load(ROOT / "data/gsm8k_en_clean.ministral.test.pt",
                    map_location="cpu", weights_only=False)
    keep = (~te["truncated"]).numpy()
    y = te["was_correct"].numpy()[keep]
    t = te["temperature"].numpy()[keep]
    X = te["hidden_frac"][:, -1, :].float().numpy()[keep]
    mu = Xe_tr.mean(0)
    s = (X - mu) @ w
    m0 = np.isclose(t, 0.0)
    thr = 0.5 * (s[m0 & y].mean() + s[m0 & ~y].mean())   # one threshold, from tau=0

    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 3.0), sharex=True)
    lo, hi = np.percentile(s, 1), np.percentile(s, 99.9)
    xs = np.linspace(lo, hi, 400)
    for ax, b in zip(axes, BANDS):
        m = np.isclose(t, float(b))
        auc = roc_auc_score(y[m], s[m])
        for cls, col, lab in [(True, GOOD, "correct"), (False, CRIT, "incorrect")]:
            v = s[m & (y == cls)]
            d = gaussian_kde(v)(xs)
            d = d / d.max()               # per-curve scaling: shapes comparable
            ax.fill_between(xs, d, color=col, alpha=0.35, lw=0)
            ax.plot(xs, d, color=col, lw=1.2,
                    label=lab if b == "0.0" else None)
        ax.axvline(thr, color=INK, lw=0.9, ls="--")
        ax.text(0.02, 0.80, f"$\\tau={b}$   AUC {auc:.2f}",
                transform=ax.transAxes, fontsize=7.5, color=INK)
        ax.set_yticks([])
        ax.set_ylim(0, 1.12)
        despine(ax)
        ax.spines["left"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right",
                   bbox_to_anchor=(1.0, 1.32))
    axes[2].set_xlabel("score under the frozen $\\tau{=}0$ arrow")
    axes[0].set_title("one threshold (dashed), fitted on greedy chains only",
                      fontsize=8, pad=16)
    save(fig, "fig_ridges")


def fig_arrow_concept():
    """Didactic mini-diagram: what the arrow is."""
    rng = np.random.default_rng(3)
    good = rng.normal([2.2, 0.4], 0.55, (110, 2))
    bad = rng.normal([0.0, -0.3], 0.62, (55, 2))
    mu_g, mu_b = good.mean(0), bad.mean(0)
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.scatter(*good.T, s=7, c=GOOD, alpha=0.4, marker="o")
    ax.scatter(*bad.T, s=10, c=CRIT, alpha=0.6, marker="x", linewidths=0.8)
    for mu, c in ((mu_g, GOOD), (mu_b, CRIT)):
        ax.scatter(*mu, s=90, c=c, edgecolors="white", linewidths=1.2, zorder=5)
    ax.annotate("", xy=mu_g, xytext=mu_b,
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=INK), zorder=4)
    mid = (mu_g + mu_b) / 2
    ax.text(mid[0] - 1.15, mid[1] + 0.8,
            "$\\mathbf{w}=\\mu_+ - \\mu_-$", fontsize=9, color=INK)
    ax.text(mu_g[0] + 1.0, mu_g[1] - 0.85, "mean of\ncorrect states",
            fontsize=6.5, color=INK, ha="center")
    ax.text(mu_b[0] - 0.55, mu_b[1] - 1.25, "mean of\nincorrect states",
            fontsize=6.5, color=INK, ha="center")
    ax.set_xlim(-2.6, 4.6)
    ax.set_ylim(-2.6, 2.6)
    d = (mu_g - mu_b) / np.linalg.norm(mu_g - mu_b)
    n = np.array([-d[1], d[0]])
    ax.plot(*np.array([mid - 2.0 * n, mid + 2.0 * n]).T,
            ls="--", lw=1.0, color=MUTED)
    ax.text(*(mid + 1.45 * n + [0.1, 0.06]), "decision\nboundary", fontsize=6.5,
            color=MUTED)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "fig_arrow_concept")


def fig_flowmap():
    """Whole-study map: data -> generation -> grading -> extraction ->
    tensors -> every analysis, each pointing to its figure/table."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def box(ax, x, y, w, h, title, body, ec="#2a78d6", fc="#eaf2fc",
            tfs=6.8, bfs=5.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.012,rounding_size=0.025",
                     fc=fc, ec=ec, lw=1.1, zorder=2))
        ax.text(x + w / 2, y + h - 0.058, title, ha="center", va="top",
                fontsize=tfs, fontweight="bold", color=INK, zorder=3)
        if body:
            ax.text(x + w / 2, y + 0.022, body, ha="center", va="bottom",
                    fontsize=bfs, color=INK, zorder=3, linespacing=1.35)

    def arrow(ax, p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                     color=MUTED, lw=1.0, zorder=1,
                     connectionstyle=f"arc3,rad={rad}"))

    fig, ax = plt.subplots(figsize=(PAGE_W, 4.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)

    # ---- tier 1: data & generation
    box(ax, 0.15, 4.55, 1.7, 1.15, "GSM8K",
        "7,473 train problems\n1,319 test problems")
    box(ax, 2.45, 4.55, 2.6, 1.15, "1. Generate  (vLLM)",
        "3 models $\\times$ 5 chains/problem\n(1 greedy, 2@$\\tau$0.6, 2@$\\tau$1.0)\n$\\approx$44k chains per model\ncollect.py --stage generate")
    box(ax, 5.65, 4.55, 1.95, 1.15, "2. Grade",
        "last '####' answer,\nexact match; drop\ntruncated chains")
    box(ax, 8.2, 4.55, 1.65, 1.15, "labels",
        "correct /\nincorrect", ec="#104281", fc="white")
    arrow(ax, (1.85, 5.12), (2.45, 5.12))
    arrow(ax, (5.05, 5.12), (5.65, 5.12))
    arrow(ax, (7.6, 5.12), (8.2, 5.12))

    # ---- tier 2: feature extraction
    box(ax, 1.3, 2.55, 4.4, 1.3, "3. Teacher-force & tap",
        "collect.py --stage extract:\n"
        "replay each chain through its own model, record residual stream at\n"
        "25 / 50 / 75 / 90 / 95 / 100% of the reasoning + pre-answer token\n"
        "+ 8 layers + per-token log-probs")
    box(ax, 6.55, 2.55, 3.3, 1.3, "feature tensors  (.pt)",
        "one row per graded chain:\nhidden states at every tap,\nlabel, temperature, length,\nlog-prob summaries", ec="#104281", fc="white")
    arrow(ax, (3.6, 4.55), (3.55, 3.85))
    arrow(ax, (8.95, 4.55), (8.6, 3.85))
    arrow(ax, (5.7, 3.2), (6.55, 3.2))

    # ---- tier 3: analyses, fed from the tensors via a horizontal bus
    Ys = 0.25
    analyses = [
        (0.15, "analyze.py", "arrow + probe AUCs,\npositional curve,\n3$\\times$3 temp transfer,\nlayers, baselines", "Tab. II, Figs. 3--5, 9"),
        (2.15, "geometry & CIs", "band-arrow cosines,\npre vs. end axis,\nbootstrap CIs", "Fig. 6"),
        (4.15, "within_prompt.py", "same-question\ncorrect-vs-incorrect\npairs", "Fig. 8"),
        (6.15, "verifier.py", "best-of-5 voting,\nrisk--coverage", "Tab. III, Fig. 10"),
        (8.15, "legacy pair data", "frozen readouts under\ndistribution shift", "Fig. 7"),
    ]
    bus_y = 2.25
    ax.plot([8.2, 8.2], [2.55, bus_y], color=MUTED, lw=1.0, zorder=1)
    ax.plot([0.15 + 0.9, 8.15 + 0.9], [bus_y, bus_y], color=MUTED, lw=1.0,
            zorder=1)
    for x, title, body, out in analyses:
        box(ax, x, Ys + 0.55, 1.8, 1.35, title, body, tfs=6.2, bfs=5.2)
        ax.text(x + 0.9, Ys + 0.28, "$\\rightarrow$ " + out, ha="center",
                fontsize=5.6, color="#104281", fontweight="bold")
        arrow(ax, (x + 0.9, bus_y), (x + 0.9, Ys + 1.95))
    save(fig, "fig_flowmap")


def fig_cascade():
    """The projection cascade: the same clouds seen in 3-D, 2-D, and finally
    as the 1-D score that decides. Columns = temperature bands; the x-axis
    (projection on the frozen tau=0 arrow) is shared by every panel."""
    tr = torch.load(ROOT / "data/gsm8k_en_clean.ministral.train.pt",
                    map_location="cpu", weights_only=False)
    te = torch.load(ROOT / "data/gsm8k_en_clean.ministral.test.pt",
                    map_location="cpu", weights_only=False)

    def end_h(d):
        return d["hidden_frac"][:, -1, :].float().numpy()

    Xtr, ytr = end_h(tr), tr["was_correct"].numpy()
    ttr, mtr = tr["temperature"].numpy(), (~tr["truncated"]).numpy()
    sel = mtr & (ttr == 0.0)
    w = Xtr[sel & ytr].mean(0) - Xtr[sel & ~ytr].mean(0)
    w /= np.linalg.norm(w)
    R = Xtr[sel] - Xtr[sel].mean(0)
    R = R - np.outer(R @ w, w)
    _, _, Vt = np.linalg.svd(R[:4000], full_matrices=False)
    p1, p2 = Vt[0], Vt[1]

    Xte, yte = end_h(te), te["was_correct"].numpy()
    tte, mte = te["temperature"].numpy(), (~te["truncated"]).numpy()
    mu = Xtr[mtr].mean(0)
    PX = (Xte[mte] - mu) @ w
    XL = (np.percentile(PX, 0.3) - 8, np.percentile(PX, 99.9) + 8)

    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(PAGE_W, 5.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 0.40],
                          hspace=0.24, wspace=0.16)

    for k, b in enumerate(BANDS):
        m = mte & np.isclose(tte, float(b))
        X, y = Xte[m] - mu, yte[m]
        px, py, pz = X @ w, X @ p1, X @ p2
        auc = roc_auc_score(y, px)
        mg = np.array([px[y].mean(), py[y].mean(), pz[y].mean()])
        mr = np.array([px[~y].mean(), py[~y].mean(), pz[~y].mean()])
        thr = 0.5 * (mg[0] + mr[0])
        keep = rng.permutation(len(y))[:900]

        # ---- row 1: 3-D, flattened into a wide slab sharing the x range
        ax = fig.add_subplot(gs[0, k], projection="3d")
        for cls, col, mk, al, z in [(True, GOOD, "o", 0.25, 2),
                                    (False, CRIT, "x", 0.7, 3)]:
            i = keep[y[keep] == cls]
            ax.scatter(px[i], py[i], pz[i], s=4 if cls else 8, marker=mk,
                       c=col, alpha=al, linewidths=0.6, zorder=z)
        ax.quiver(*mr, *(mg - mr), color=INK, lw=2.0,
                  arrow_length_ratio=0.12, zorder=6)
        ax.scatter(*mr, color=CRIT, s=45, edgecolors="white", lw=1.0, zorder=7)
        ax.scatter(*mg, color=GOOD, s=45, edgecolors="white", lw=1.0, zorder=7)
        ax.set_proj_type("ortho")
        ax.view_init(elev=14, azim=-80)
        ax.set_xlim(*XL)
        ax.set_ylim(np.percentile(py, 0.3), np.percentile(py, 99.9))
        ax.set_zlim(np.percentile(pz, 0.3), np.percentile(pz, 99.9))
        ax.set_box_aspect((2.5, 1.0, 0.85))
        ax.set_title(f"$\\tau={b}$    AUC {auc:.2f}", fontsize=8.5, pad=0)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_ticklabels([])
            pane.pane.set_alpha(0.04)
            pane.line.set_color(BASE)
        ax.grid(False)
        if k == 0:
            ax.text2D(-0.08, 0.5, "3-D view\n(two nuisance axes)",
                      transform=ax.transAxes, rotation=90, va="center",
                      ha="center", fontsize=7.5, color=INK)

        # ---- row 2: 2-D
        ax = fig.add_subplot(gs[1, k])
        for cls, col, mk, al, z in [(True, GOOD, "o", 0.3, 2),
                                    (False, CRIT, "x", 0.75, 3)]:
            i = keep[y[keep] == cls]
            ax.scatter(px[i], py[i], s=5 if cls else 9, marker=mk, c=col,
                       alpha=al, linewidths=0.7, zorder=z)
        ax.annotate("", xy=(mg[0], mg[1]), xytext=(mr[0], mr[1]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=INK),
                    zorder=6)
        ax.scatter([mr[0], mg[0]], [mr[1], mg[1]], c=[CRIT, GOOD], s=40,
                   edgecolors="white", lw=1.0, zorder=7)
        ax.axvline(thr, color=INK, lw=0.8, ls="--")
        ax.set_xlim(*XL)
        ax.set_xticklabels([])
        despine(ax)
        if k == 0:
            ax.set_ylabel("2-D view\n(one nuisance axis)", fontsize=7.5)
        else:
            ax.set_yticklabels([])

        # ---- row 3: 1-D (the score itself)
        ax = fig.add_subplot(gs[2, k])
        jit = rng.normal(0, 0.16, len(px))
        for cls, col, mk, al, z in [(True, GOOD, "o", 0.25, 2),
                                    (False, CRIT, "x", 0.7, 3)]:
            i = keep[y[keep] == cls]
            ax.scatter(px[i], jit[i], s=4 if cls else 8, marker=mk, c=col,
                       alpha=al, linewidths=0.6, zorder=z)
        ax.annotate("", xy=(mg[0], 0.78), xytext=(mr[0], 0.78),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=INK),
                    zorder=6)
        ax.scatter([mr[0], mg[0]], [0.78, 0.78], c=[CRIT, GOOD], s=32,
                   edgecolors="white", lw=0.9, zorder=7)
        ax.axvline(thr, color=INK, lw=0.9, ls="--")
        ax.text(thr - 4, 1.02, "midpoint", ha="right", fontsize=6.2, color=INK)
        ax.set_xlim(*XL)
        ax.set_ylim(-1.15, 1.45)
        ax.set_yticks([])
        despine(ax)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("projection on frozen $\\tau{=}0$ arrow", fontsize=7.5)
        if k == 0:
            ax.set_ylabel("1-D:\nthe score", fontsize=7.5)

    handles = [plt.Line2D([], [], marker="o", ls="", color=GOOD, label="correct"),
               plt.Line2D([], [], marker="x", ls="", color=CRIT, label="incorrect"),
               plt.Line2D([], [], marker=r"$\rightarrow$", ls="", color=INK,
                          markersize=12, label="arrow (red mean $\\to$ green mean)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.015))
    save(fig, "fig_cascade")
