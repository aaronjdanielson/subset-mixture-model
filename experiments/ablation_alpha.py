"""
Ablation study: effect of Dirichlet concentration parameter alpha.

Runs SMM on all four datasets with alpha in {1.01, 1.1, 1.5, 2.0, 5.0}
and reports RMSE, NLL, Coverage, and weight entropy (how concentrated
the learned weights are).

Run from smm_paper/ root:
    python experiments/ablation_alpha.py
"""

import sys, pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import entropy as scipy_entropy

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smm import (SubsetMaker, SubsetWeightsModel, SubsetDataset,
                 subset_mixture_neg_log_posterior, SubsetMixturePredictor,
                 compute_posterior_covariance, predict_with_uncertainty, coverage)
from experiments.run_experiments import (
    DATASETS, HP, rmse, mae, gaussian_nll
)

ALPHAS = [1.01, 1.1, 1.5, 2.0, 5.0]


def train_and_eval(train_df, val_df, test_df, cat_cols, target, alpha, ds_name):
    subset_maker = SubsetMaker(train_df, cat_cols, [target])
    powerset = list(subset_maker.lookup.keys())
    num_subsets = len(powerset)

    train_ds = SubsetDataset(train_df, cat_cols, [target])
    val_ds   = SubsetDataset(val_df,   cat_cols, [target])
    train_loader = DataLoader(train_ds, batch_size=HP["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=HP["batch_size"], shuffle=False)

    model = SubsetWeightsModel(num_subsets)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HP["lr"], weight_decay=HP["weight_decay"])

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(HP["num_epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pi_logits = model()
            mus, variances, mask = subset_maker.batch_lookup(x)
            loss = subset_mixture_neg_log_posterior(
                pi_logits, y, mus, variances, mask, alpha=alpha)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                mus, variances, mask = subset_maker.batch_lookup(x)
                val_loss += subset_mixture_neg_log_posterior(
                    model(), y, mus, variances, mask, alpha=alpha).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= HP["patience"] and epoch >= HP["min_epochs"]:
            break

    model.load_state_dict(best_state)
    pi_hat = F.softmax(model.eta.detach(), dim=0)
    predictor = SubsetMixturePredictor(subset_maker, pi_hat)

    sigma_pi = compute_posterior_covariance(
        model, subset_maker, train_df, cat_cols, target, alpha=alpha)
    y_mean, y_std = predict_with_uncertainty(predictor, sigma_pi, test_df)

    y_te = test_df[target].values
    cov95 = coverage(y_te, y_mean, y_std)
    ent = float(scipy_entropy(pi_hat.numpy()))   # entropy of weight distribution

    return {
        "alpha":   alpha,
        "dataset": ds_name,
        "RMSE":    rmse(y_te, y_mean),
        "NLL":     gaussian_nll(y_te, y_mean, y_std),
        "Cov95":   cov95,
        "Entropy": ent,
    }


def main():
    rows = []
    for ds_name, cfg in DATASETS.items():
        data_dir = ROOT / "data" / ds_name
        if not data_dir.exists():
            print(f"Skipping {ds_name}: data not found")
            continue

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

        print(f"\n=== {ds_name} ===")
        for alpha in ALPHAS:
            print(f"  alpha={alpha}", end="", flush=True)
            result = train_and_eval(
                train_df, val_df, test_df, cat_cols, target, alpha, ds_name)
            rows.append(result)
            print(f"  RMSE={result['RMSE']:.4f}  NLL={result['NLL']:.4f}"
                  f"  Cov95={result['Cov95']:.3f}  Entropy={result['Entropy']:.3f}")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "ablation_alpha.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
