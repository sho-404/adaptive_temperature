"""Train the adaptive-temperature scorer for ONE language (LOCAL, no GPU).

This is a conditional Bradley-Terry / LPO scorer. It learns a function

    score(prompt_features, temperature) -> scalar

and at inference picks  argmax_tau score(f, tau)  over the temperature grid.
Training uses the LPO loss (paper Eq. 10, "temperatures as tokens, separate"),
which with a uniform reference collapses to a logistic loss on the score gap:

    L = - log sigmoid( beta * ( score(f, tau_chosen) - score(f, tau_rejected) ) )

Data reality for GSM8K (see header diagnostics printed at runtime):
  - easy pairs (~85%): chosen tau is ALWAYS 0.0 (greedy), rejected is higher.
  - hard pairs (~15%): chosen tau is ALWAYS higher than rejected.
  - all_fail pairs: DROPPED (no correctness signal).

Therefore the trivial rule "always prefer the lower temperature" scores ~86.7%
WITHOUT using any features. A LINEAR model on [features, tau] is provably equal
to that trivial rule (the feature term cancels in the score gap). Only a model
with a feature x temperature INTERACTION (the MLP) can use the prompt to beat it
-- i.e. learn to predict, from the frozen prompt features, which prompts need a
higher temperature. That gap above 86.7% (and the hard-pair accuracy in
particular) is the real measure of learning.

Runs locally off features/gsm8k_<lang>.pt + datasets/gsm8k_<lang>.jsonl.

Usage:
    python train_mlp.py --lang en                       # MLP, alpha=1
    python train_mlp.py --lang en --alpha 8             # upweight hard pairs
    python train_mlp.py --lang en --linear              # linear control (==trivial bar)
    python train_mlp.py --lang en --no-hidden           # ablate S3 (scalars only)
    python train_mlp.py --lang en --no-s4               # ablate the suspect feature
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).parents[2] / "data"


# ============================================================
# Data
# ============================================================

def load_pairs(lang: str) -> list[dict]:
    path = DATA / f"gsm8k_{lang}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(lang: str, use: dict) -> dict:
    """Join pairs <-> features by index, drop all_fail, assemble tensors.

    Returns per-pair tensors: feat [N,D], tau_c [N], tau_r [N], is_hard [N],
    plus the standardisation stats and the temperature grid.
    """
    feats = torch.load(DATA / f"gsm8k_{lang}.pt", map_location="cpu", weights_only=False)
    idx_to_row = {int(i): k for k, i in enumerate(feats["index"].tolist())}

    pairs = load_pairs(lang)
    grid = sorted({p["chosen"]["temperature"] for p in pairs} |
                  {p["rejected"]["temperature"] for p in pairs})

    feat_rows, tau_c, tau_r, is_hard = [], [], [], []
    dropped_all_fail = 0
    for p in pairs:
        if p["mode"] == "all_fail":
            dropped_all_fail += 1
            continue
        r = idx_to_row[int(p["index"])]

        cols = []
        if use["hidden"]:
            cols.append(feats["hidden"][r])                      # [3072]
        if use["s1"]:
            cols.append(feats["entropy"][r].reshape(1))
        if use["s2"]:
            cols.append(feats["logit_gap"][r].reshape(1))
        if use["s4"]:
            cols.append(feats["attn_entropy"][r].reshape(1))
        feat_rows.append(torch.cat(cols).float())

        tau_c.append(p["chosen"]["temperature"])
        tau_r.append(p["rejected"]["temperature"])
        is_hard.append(p["mode"] == "hard")

    return {
        "feat": torch.stack(feat_rows),
        "tau_c": torch.tensor(tau_c, dtype=torch.float32),
        "tau_r": torch.tensor(tau_r, dtype=torch.float32),
        "is_hard": torch.tensor(is_hard, dtype=torch.bool),
        "grid": torch.tensor(grid, dtype=torch.float32),
        "dropped_all_fail": dropped_all_fail,
    }


def split_by_prompt(n: int, is_hard: torch.Tensor, val_frac: float, seed: int):
    """Stratified train/val split BY PROMPT (each pair is one prompt here).

    Stratified on mode so val has a representative easy/hard mix; never mixes a
    prompt across splits (each row is a distinct prompt, so this is automatic).
    """
    g = torch.Generator().manual_seed(seed)
    val_mask = torch.zeros(n, dtype=torch.bool)
    for cls in (False, True):
        ids = torch.nonzero(is_hard == cls).squeeze(1)
        perm = ids[torch.randperm(len(ids), generator=g)]
        n_val = int(round(len(ids) * val_frac))
        val_mask[perm[:n_val]] = True
    return ~val_mask, val_mask


# ============================================================
# Model
# ============================================================

class Scorer(nn.Module):
    """score(features, tau) -> scalar. tau is appended as an input feature."""

    def __init__(self, d_feat: int, hidden_dims: list[int], dropout: float, linear: bool):
        super().__init__()
        d_in = d_feat + 1  # + tau
        if linear:
            self.net = nn.Linear(d_in, 1)
            return
        layers, d = [], d_in
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, tau.reshape(-1, 1)], dim=1)
        return self.net(x).squeeze(1)


# ============================================================
# Eval helpers
# ============================================================

@torch.no_grad()
def pair_accuracy(model, feat, tau_c, tau_r, is_hard) -> dict:
    """Fraction of pairs where score(chosen) > score(rejected), by subset."""
    model.eval()
    s_c = model(feat, tau_c)
    s_r = model(feat, tau_r)
    correct = (s_c > s_r)
    out = {"overall": correct.float().mean().item()}
    out["easy"] = correct[~is_hard].float().mean().item() if (~is_hard).any() else float("nan")
    out["hard"] = correct[is_hard].float().mean().item() if is_hard.any() else float("nan")
    return out


def trivial_lower_temp_accuracy(tau_c, tau_r, is_hard) -> dict:
    """Bar to beat: always prefer the LOWER temperature. Uses zero features."""
    correct = (tau_c < tau_r)
    return {
        "overall": correct.float().mean().item(),
        "easy": correct[~is_hard].float().mean().item(),
        "hard": correct[is_hard].float().mean().item(),
    }


@torch.no_grad()
def predicted_temp_distribution(model, feat, grid) -> dict:
    """For each prompt, argmax score over the grid -> distribution (collapse check)."""
    model.eval()
    scores = torch.stack([model(feat, torch.full((feat.shape[0],), t)) for t in grid.tolist()], dim=1)
    picks = grid[scores.argmax(dim=1)]
    return {round(t, 2): int((picks == t).sum()) for t in grid.tolist()}


# ============================================================
# Train
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight on HARD pairs (easy=1.0)")
    ap.add_argument("--beta", type=float, default=1.0, help="LPO/DPO logit scale")
    ap.add_argument("--linear", action="store_true", help="linear control model (== trivial bar)")
    ap.add_argument("--no-hidden", dest="hidden", action="store_false", help="ablate S3 hidden state")
    ap.add_argument("--no-s1", dest="s1", action="store_false")
    ap.add_argument("--no-s2", dest="s2", action="store_false")
    ap.add_argument("--no-s4", dest="s4", action="store_false", help="ablate the suspect attention feature")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25, help="early-stop on val overall acc")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="write checkpoints/mlp_<lang>.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    use = {"hidden": args.hidden, "s1": args.s1, "s2": args.s2, "s4": args.s4}

    ds = build_dataset(args.lang, use)
    feat, tau_c, tau_r, is_hard = ds["feat"], ds["tau_c"], ds["tau_r"], ds["is_hard"]
    grid, n = ds["grid"], feat.shape[0]

    tr, va = split_by_prompt(n, is_hard, args.val_frac, args.seed)

    # Standardise features on TRAIN stats only.
    mu = feat[tr].mean(0, keepdim=True)
    sd = feat[tr].std(0, keepdim=True).clamp_min(1e-6)
    feat = (feat - mu) / sd

    active = [k.upper().replace("HIDDEN", "S3") for k, v in use.items() if v]
    print("=" * 64)
    print(f"Adaptive-temperature scorer  |  lang={args.lang}")
    print(f"  pairs used     : {n}  (dropped {ds['dropped_all_fail']} all_fail)")
    print(f"  temp grid      : {[round(t,2) for t in grid.tolist()]}")
    print(f"  features       : {active}  -> D={feat.shape[1]}")
    print(f"  model          : {'LINEAR (control)' if args.linear else 'MLP ' + str(args.hidden_dims)}")
    print(f"  alpha (hard)   : {args.alpha}   beta: {args.beta}")
    print(f"  train / val    : {int(tr.sum())} / {int(va.sum())}  "
          f"(val hard={int(is_hard[va].sum())}, easy={int((~is_hard[va]).sum())})")

    base = trivial_lower_temp_accuracy(tau_c[va], tau_r[va], is_hard[va])
    print(f"  >> BAR TO BEAT (prefer-lower-temp, no features): "
          f"overall={base['overall']:.3f}  easy={base['easy']:.3f}  hard={base['hard']:.3f}")
    print("=" * 64)

    model = Scorer(feat.shape[1], args.hidden_dims, args.dropout, args.linear)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    w = torch.where(is_hard, torch.tensor(args.alpha), torch.tensor(1.0))
    f_tr, c_tr, r_tr, w_tr = feat[tr], tau_c[tr], tau_r[tr], w[tr]

    best = {"overall": -1.0}
    best_state, best_epoch, since = None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        gap = model(f_tr, c_tr) - model(f_tr, r_tr)
        loss = -(w_tr * F.logsigmoid(args.beta * gap)).sum() / w_tr.sum()
        loss.backward()
        opt.step()

        val = pair_accuracy(model, feat[va], tau_c[va], tau_r[va], is_hard[va])
        if val["overall"] > best["overall"]:
            best = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch, since = epoch, 0
        else:
            since += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  "
                  f"val overall {val['overall']:.3f}  easy {val['easy']:.3f}  hard {val['hard']:.3f}")
        if since >= args.patience:
            print(f"  early stop @ epoch {epoch} (no val gain for {args.patience})")
            break

    model.load_state_dict(best_state)
    tr_acc = pair_accuracy(model, feat[tr], tau_c[tr], tau_r[tr], is_hard[tr])
    dist = predicted_temp_distribution(model, feat[va], grid)

    print("-" * 64)
    print(f"BEST (epoch {best_epoch})")
    print(f"  VAL   overall {best['overall']:.3f}  easy {best['easy']:.3f}  hard {best['hard']:.3f}")
    print(f"  TRAIN overall {tr_acc['overall']:.3f}  easy {tr_acc['easy']:.3f}  hard {tr_acc['hard']:.3f}   "
          f"(train-val overall gap = {tr_acc['overall']-best['overall']:+.3f})")
    print(f"  delta vs bar : overall {best['overall']-base['overall']:+.3f}   "
          f"hard {best['hard']-base['hard']:+.3f}")
    print(f"  predicted-temp distribution on val (collapse check): {dist}")
    print("-" * 64)

    if args.save:
        ckpt_dir = DATA / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                    "grid": grid, "use": use, "args": vars(args)},
                   ckpt_dir / f"mlp_{args.lang}.pt")
        print(f"saved -> checkpoints/mlp_{args.lang}.pt")


if __name__ == "__main__":
    main()
