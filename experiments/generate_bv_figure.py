"""
Generate bias-variance tradeoff figure for the SMM paper.

Shows, for the NBA dataset, how average cell size and total learned weight
vary with subset order k. Visually explains why higher-order subsets can
hurt when cells are sparse, and why the mixture provides a principled
bias-variance tradeoff.

Run from smm_paper/ root:
    python experiments/generate_bv_figure.py

Output: results/figures/bias_variance.pdf
"""

import sys, pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smm import (SubsetMaker, SubsetWeightsModel, SubsetDataset,
                 subset_mixture_neg_log_posterior)
from experiments.run_experiments import DATASETS, HP

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})

FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def train_smm_nba(tr, va, cat_cols, target, seed=42):
    torch.manual_seed(seed)
    subset_maker = SubsetMaker(tr, cat_cols, [target])
    powerset = list(subset_maker.lookup.keys())   # list of tuples of feature names
    model = SubsetWeightsModel(len(powerset))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HP["lr"], weight_decay=HP["weight_decay"])
    train_loader = DataLoader(SubsetDataset(tr, cat_cols, [target]),
                              batch_size=HP["batch_size"], shuffle=True)
    val_loader   = DataLoader(SubsetDataset(va, cat_cols, [target]),
                              batch_size=HP["batch_size"], shuffle=False)
    best_val, no_imp, best_state = float("inf"), 0, None
    for epoch in range(HP["num_epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            mus, variances, mask = subset_maker.batch_lookup(x)
            subset_mixture_neg_log_posterior(
                model(), y, mus, variances, mask, alpha=HP["alpha"]).backward()
            optimizer.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                mus, variances, mask = subset_maker.batch_lookup(x)
                vl += subset_mixture_neg_log_posterior(
                    model(), y, mus, variances, mask, alpha=HP["alpha"]).item()
        vl /= len(val_loader)
        if vl < best_val:
            best_val, no_imp = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= HP["patience"] and epoch >= HP["min_epochs"]:
            break
    model.load_state_dict(best_state)
    pi_hat = F.softmax(model.eta.detach(), dim=0)
    return subset_maker, pi_hat, powerset


def main():
    cfg = DATASETS["nba"]
    data_dir = ROOT / "data" / "nba"
    cat_cols = cfg["cat_cols"]
    target   = cfg["target"]
    keep     = cat_cols + [target]

    tr = pd.read_csv(data_dir / "train.csv")[keep].dropna().copy()
    va = pd.read_csv(data_dir / cfg["val_file"])[keep].dropna().copy()

    # Encode string columns using training categories
    for col in cat_cols:
        if not pd.api.types.is_numeric_dtype(tr[col]):
            cats = pd.CategoricalDtype(
                categories=tr[col].astype("category").cat.categories)
            tr[col] = tr[col].astype(cats).cat.codes.astype(int)
            va[col] = va[col].astype(cats).cat.codes.astype(int)

    print("Training SMM on NBA...")
    subset_maker, pi_hat, powerset = train_smm_nba(tr, va, cat_cols, target)
    pi_arr = pi_hat.numpy()

    # Aggregate statistics by subset order k
    k_vals = sorted(set(len(s) for s in powerset))
    k_mean_n   = []   # average number of training examples per non-empty cell
    k_total_w  = []   # total learned weight assigned to subsets of order k
    k_n_cells  = []   # average number of distinct cells (cardinality)
    k_plug_var = []   # total plug-in variance contribution sum_s pi_s^2 * mean(sigma^2/n)

    for k in k_vals:
        subsets_k = [(i, s) for i, s in enumerate(powerset) if len(s) == k]
        total_w = sum(float(pi_arr[i]) for i, _ in subsets_k)
        k_total_w.append(total_w)

        # Compute cell sizes and variance from training data
        all_mean_n = []
        all_vc = []
        for i, s in subsets_k:
            grp = tr.groupby(list(s))[target]
            cell_n    = grp.count().values.astype(float)
            cell_var  = grp.var(ddof=1).fillna(0).values
            # keep non-empty cells
            mask = cell_n > 1
            if mask.sum() == 0:
                continue
            n_arr  = cell_n[mask]
            v_arr  = cell_var[mask]
            all_mean_n.append(n_arr.mean())
            pi_s = float(pi_arr[i])
            vc = pi_s**2 * float((v_arr / n_arr).mean())
            all_vc.append(vc)

        k_mean_n.append(np.mean(all_mean_n) if all_mean_n else 0)
        k_plug_var.append(sum(all_vc))

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    fig.suptitle("Bias–Variance Tradeoff by Subset Order (NBA Attendance)", y=1.02)

    ks = k_vals
    bar_color = "#4C72B0"
    red_color = "#C44E52"

    # Panel 1: mean cell size per order
    axes[0].bar(ks, k_mean_n, color=bar_color, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Subset order $k$")
    axes[0].set_ylabel("Mean cell size $\\bar{n}_k$")
    axes[0].set_title("(a) Cell size decreases\nwith order")
    axes[0].set_xticks(ks)
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Panel 2: total learned weight per order
    axes[1].bar(ks, k_total_w, color=bar_color, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Subset order $k$")
    axes[1].set_ylabel(r"$\sum_{|s|=k}\hat{\pi}_s$")
    axes[1].set_title("(b) SMM up-weights $k=3$\n(best bias-variance point)")
    axes[1].set_xticks(ks)
    # Highlight k=3 bar
    axes[1].bar([3], [k_total_w[2]], color=red_color, alpha=0.85, edgecolor="white")

    # Panel 3: plug-in variance contribution per order
    axes[2].bar(ks, k_plug_var, color=red_color, alpha=0.85, edgecolor="white")
    axes[2].set_xlabel("Subset order $k$")
    axes[2].set_ylabel(r"$\sum_{|s|=k}\hat{\pi}_s^2\cdot\overline{\sigma^2/n_s}$")
    axes[2].set_title(r"(c) Plug-in variance $V_\mathrm{plug}$" + "\nspikes at high $k$")
    axes[2].set_xticks(ks)
    axes[2].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))

    plt.tight_layout()
    out = FIGURES_DIR / "bias_variance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    print("\nOrder  MeanCellN  TotalWeight  VarContrib")
    for k, n, w, vc in zip(ks, k_mean_n, k_total_w, k_plug_var):
        print(f"  k={k}    {n:8.1f}    {w:.4f}       {vc:.6f}")


if __name__ == "__main__":
    main()
