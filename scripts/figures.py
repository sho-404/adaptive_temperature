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
