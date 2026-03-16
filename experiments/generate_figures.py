"""
Generate publication-quality figures for the SMM paper.

Run from smm_paper/ root:
    python experiments/generate_figures.py

Produces:
  results/figures/weight_bars_<dataset>.pdf   -- top-10 subset weights (all 4 datasets)
  results/figures/calibration_curve.pdf       -- reliability diagram (SMM vs Bayesian Ridge)
  results/figures/nba_scatter.pdf             -- predicted vs actual (NBA, SMM vs LightGBM)
"""

import sys, pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from scipy.stats import norm as scipy_norm

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smm import (SubsetMaker, SubsetWeightsModel, SubsetDataset,
                 subset_mixture_neg_log_posterior, SubsetMixturePredictor,
                 compute_posterior_covariance, predict_with_uncertainty)
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder
from experiments.run_experiments import DATASETS, HP

FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})

DATASET_LABELS = {
    "nba":                 "NBA Attendance",
    "bike_sharing":        "Bike Sharing",
    "student_performance": "Student Performance",
    "ames_housing":        "Ames Housing",
}

SUBSET_PRETTY = {
    "home_team": "Home", "visitor_team": "Visitor", "arena_name": "Arena",
    "start_hour": "Hour", "day_of_week": "Day", "month": "Month",
    "season": "Season", "holiday": "Holiday", "weekday": "Weekday",
    "weathersit": "Weather", "workingday": "Workday",
    "school": "School", "sex": "Sex", "address": "Address",
    "Mjob": "Mother job", "Fjob": "Father job",
    "Neighborhood": "Neighborhood", "BldgType": "Bldg type",
    "HouseStyle": "Style", "Foundation": "Foundation", "GarageType": "Garage",
}


def pretty_subset(subset_tuple):
    return " + ".join(SUBSET_PRETTY.get(f, f) for f in subset_tuple)


def load_and_encode(ds_name, cfg):
    data_dir = ROOT / "data" / ds_name
    cat_cols = cfg["cat_cols"]
    target   = cfg["target"]
    keep     = cat_cols + [target]

    train_df = pd.read_csv(data_dir / "train.csv")[keep].dropna().copy()
    val_df   = pd.read_csv(data_dir / cfg["val_file"])[keep].dropna().copy()
    test_df  = pd.read_csv(data_dir / "test.csv")[keep].dropna().copy()

    for col in cat_cols:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            cats = pd.CategoricalDtype(
                categories=train_df[col].astype("category").cat.categories)
            train_df[col] = train_df[col].astype(cats).cat.codes.astype(int)
            val_df[col]   = val_df[col].astype(cats).cat.codes.astype(int)
            test_df[col]  = test_df[col].astype(cats).cat.codes.astype(int)

    return train_df, val_df, test_df


def train_smm(train_df, val_df, cat_cols, target, alpha=1.1):
    subset_maker = SubsetMaker(train_df, cat_cols, [target])
    powerset     = list(subset_maker.lookup.keys())
    model        = SubsetWeightsModel(len(powerset))
    optimizer    = torch.optim.Adam(model.parameters(), lr=HP["lr"],
                                    weight_decay=HP["weight_decay"])
    train_loader = DataLoader(SubsetDataset(train_df, cat_cols, [target]),
                              batch_size=HP["batch_size"], shuffle=True)
    val_loader   = DataLoader(SubsetDataset(val_df,   cat_cols, [target]),
                              batch_size=HP["batch_size"], shuffle=False)

    best_val, no_improve, best_state = float("inf"), 0, None
    for epoch in range(HP["num_epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            mus, variances, mask = subset_maker.batch_lookup(x)
            subset_mixture_neg_log_posterior(
                model(), y, mus, variances, mask, alpha=alpha).backward()
            optimizer.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                mus, variances, mask = subset_maker.batch_lookup(x)
                vl += subset_mixture_neg_log_posterior(
                    model(), y, mus, variances, mask, alpha=alpha).item()
        vl /= len(val_loader)
        if vl < best_val:
            best_val, no_improve = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= HP["patience"] and epoch >= HP["min_epochs"]:
            break

    model.load_state_dict(best_state)
    pi_hat    = F.softmax(model.eta.detach(), dim=0)
    predictor = SubsetMixturePredictor(subset_maker, pi_hat)
    sigma_pi  = compute_posterior_covariance(
        model, subset_maker, train_df, cat_cols, target, alpha=alpha)
    return predictor, sigma_pi, model, subset_maker, pi_hat, powerset


# ─────────────────────────────────────────────────────────────
# Figure 1: Weight bar charts (2×2 grid)
# ─────────────────────────────────────────────────────────────

def figure_weight_bars():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()

    for ax, (ds_name, cfg) in zip(axes, DATASETS.items()):
        data_dir = ROOT / "data" / ds_name
        if not data_dir.exists():
            ax.set_visible(False)
            continue

        cat_cols = cfg["cat_cols"]
        target   = cfg["target"]
        train_df, val_df, _ = load_and_encode(ds_name, cfg)

        print(f"  Training SMM for {ds_name} weight chart...")
        _, _, _, _, pi_hat, powerset = train_smm(train_df, val_df, cat_cols, target)

        top_n   = min(10, len(pi_hat))
        top_idx = torch.argsort(pi_hat, descending=True)[:top_n]
        labels  = [pretty_subset(powerset[i]) for i in top_idx]
        vals    = pi_hat[top_idx].numpy()

        bars = ax.barh(range(top_n), vals[::-1], color="#2166ac", alpha=0.85,
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel("Mixture weight", fontsize=9)
        ax.set_title(DATASET_LABELS[ds_name], fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate top bar
        ax.text(vals[0] + 0.002, top_n - 1,
                f"{vals[0]:.1%}", va="center", fontsize=7, color="#2166ac")

    fig.suptitle("Learned Mixture Weights by Feature Subset",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "weight_bars.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────
# Figure 2: NBA weight bars (standalone, larger)
# ─────────────────────────────────────────────────────────────

def figure_nba_weights():
    cfg      = DATASETS["nba"]
    cat_cols = cfg["cat_cols"]
    target   = cfg["target"]
    train_df, val_df, _ = load_and_encode("nba", cfg)

    print("  Training SMM for standalone NBA weight chart...")
    _, _, _, _, pi_hat, powerset = train_smm(train_df, val_df, cat_cols, target)

    top_n   = 10
    top_idx = torch.argsort(pi_hat, descending=True)[:top_n]
    labels  = [pretty_subset(powerset[i]) for i in top_idx]
    vals    = pi_hat[top_idx].numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#2166ac" if i == 0 else "#6baed6" for i in range(top_n)]
    ax.barh(range(top_n), vals[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Mixture weight $\\hat{\\pi}_s$")
    ax.set_title("NBA Attendance: Top-10 Learned Subset Weights", fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)

    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.001, i, f"{v:.1%}", va="center", fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "nba_weights.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────
# Figure 3: Calibration reliability diagram
# ─────────────────────────────────────────────────────────────

def figure_calibration():
    levels  = np.linspace(0.05, 0.99, 30)
    ds_list = ["nba", "bike_sharing", "student_performance", "ames_housing"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)

    for ax, ds_name in zip(axes, ds_list):
        cfg      = DATASETS[ds_name]
        cat_cols = cfg["cat_cols"]
        target   = cfg["target"]
        train_df, val_df, test_df = load_and_encode(ds_name, cfg)
        y_te = test_df[target].values

        print(f"  Calibration: training models for {ds_name}...")

        # SMM
        predictor, sigma_pi, *_ = train_smm(train_df, val_df, cat_cols, target)
        y_mean_smm, y_std_smm   = predict_with_uncertainty(
            predictor, sigma_pi, test_df)

        # Bayesian Ridge
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr = enc.fit_transform(train_df[cat_cols].astype(str))
        X_te = enc.transform(test_df[cat_cols].astype(str))
        br   = BayesianRidge().fit(X_tr, train_df[target].values)
        y_mean_br, y_std_br = br.predict(X_te, return_std=True)

        def empirical_coverage(y, mu, sigma, lvls):
            return [
                float(((y >= mu - scipy_norm.ppf((1+l)/2)*sigma) &
                       (y <= mu + scipy_norm.ppf((1+l)/2)*sigma)).mean())
                for l in lvls
            ]

        cov_smm = empirical_coverage(y_te, y_mean_smm, y_std_smm, levels)
        cov_br  = empirical_coverage(y_te, y_mean_br,  y_std_br,  levels)

        ax.plot(levels, levels,       "k--", lw=1,   label="Perfect", alpha=0.5)
        ax.plot(levels, cov_smm, "-", color="#2166ac", lw=2, label="SMM")
        ax.plot(levels, cov_br,  "-", color="#d6604d", lw=2, label="Bayes Ridge")
        ax.set_title(DATASET_LABELS[ds_name], fontsize=10, fontweight="bold")
        ax.set_xlabel("Nominal coverage")
        if ax == axes[0]:
            ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0.05, 0.99); ax.set_ylim(0.0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("Calibration: Empirical vs.\ Nominal Coverage",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "calibration_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────
# Figure 4: NBA scatter — SMM vs LightGBM
# ─────────────────────────────────────────────────────────────

def figure_nba_scatter():
    import lightgbm as lgb
    cfg      = DATASETS["nba"]
    cat_cols = cfg["cat_cols"]
    target   = cfg["target"]
    train_df, val_df, test_df = load_and_encode("nba", cfg)
    y_te = test_df[target].values

    print("  NBA scatter: training models...")
    predictor, sigma_pi, *_ = train_smm(train_df, val_df, cat_cols, target)
    y_smm, _ = predict_with_uncertainty(predictor, sigma_pi, test_df)

    enc  = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(train_df[cat_cols].astype(str))
    X_te = enc.transform(test_df[cat_cols].astype(str))
    lgbm = lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                              random_state=42, verbose=-1)
    lgbm.fit(X_tr, train_df[target].values)
    y_lgbm = lgbm.predict(X_te)

    lims = (min(y_te.min(), y_smm.min(), y_lgbm.min()) - 500,
            max(y_te.max(), y_smm.max(), y_lgbm.max()) + 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    kw = dict(alpha=0.35, s=12, rasterized=True)
    ax1.scatter(y_te, y_smm,  color="#2166ac", **kw)
    ax2.scatter(y_te, y_lgbm, color="#d6604d", **kw)

    for ax, name, rmse_val in zip(
        [ax1, ax2], ["SMM (ours)", "LightGBM"],
        [np.sqrt(np.mean((y_te - y_smm)**2)),
         np.sqrt(np.mean((y_te - y_lgbm)**2))]
    ):
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True attendance")
        ax.set_title(f"{name}\nRMSE = {rmse_val:,.0f}", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k"))

    ax1.set_ylabel("Predicted attendance")
    fig.suptitle("NBA Attendance: Predicted vs.\ Actual",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "nba_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("=== Generating figures ===")
    print("\n[1/4] Weight bar charts (all datasets)...")
    figure_weight_bars()
    print("\n[2/4] NBA standalone weight chart...")
    figure_nba_weights()
    print("\n[3/4] Calibration reliability diagrams...")
    figure_calibration()
    print("\n[4/4] NBA scatter plot...")
    figure_nba_scatter()
    print("\nAll figures saved to results/figures/")
