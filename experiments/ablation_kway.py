"""
Ablation: effect of maximum subset order k on SMM performance.

Trains SMM restricted to subsets of size <= k for k in {1, 2, 3, full}
on all six datasets. Measures whether using the full powerset is necessary
or whether low-order interactions suffice.

Run from smm_paper/ root:
    python experiments/ablation_kway.py
"""

import sys, pathlib, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smm import (SubsetMaker, SubsetWeightsModel, SubsetDataset,
                 subset_mixture_neg_log_posterior, SubsetMixturePredictor,
                 compute_posterior_covariance, predict_with_uncertainty, coverage)
from experiments.run_experiments import DATASETS, HP, rmse, mae, gaussian_nll


class SubsetMakerKWay(SubsetMaker):
    """SubsetMaker restricted to subsets of size at most max_order."""
    def __init__(self, df, subset_features, target, max_order=None):
        self.max_order = max_order if max_order is not None else len(subset_features)
        super().__init__(df, subset_features, target)

    def _get_powerset(self):
        powerset = []
        for r in range(1, self.max_order + 1):
            powerset += list(itertools.combinations(self.subset_features, r))
        return [list(s) for s in powerset]


def train_eval(train_df, val_df, test_df, cat_cols, target, max_order):
    subset_maker = SubsetMakerKWay(train_df, cat_cols, [target],
                                    max_order=max_order)
    powerset = list(subset_maker.lookup.keys())
    num_subsets = len(powerset)

    model = SubsetWeightsModel(num_subsets)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=HP["lr"], weight_decay=HP["weight_decay"])
    train_loader = DataLoader(SubsetDataset(train_df, cat_cols, [target]),
                               batch_size=HP["batch_size"], shuffle=True)
    val_loader   = DataLoader(SubsetDataset(val_df, cat_cols, [target]),
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
    pi_hat    = F.softmax(model.eta.detach(), dim=0)
    predictor = SubsetMixturePredictor(subset_maker, pi_hat)
    sigma_pi  = compute_posterior_covariance(
        model, subset_maker, train_df, cat_cols, target, alpha=HP["alpha"])
    y_mean, y_std = predict_with_uncertainty(predictor, sigma_pi, test_df)
    y_te = test_df[target].values
    return {
        "dataset":    None,
        "max_order":  max_order,
        "num_subsets": num_subsets,
        "RMSE":   rmse(y_te, y_mean),
        "NLL":    gaussian_nll(y_te, y_mean, y_std),
        "Cov95":  float(coverage(y_te, y_mean, y_std, level=0.95)),
    }


def main():
    rows = []
    for ds_name, cfg in DATASETS.items():
        data_dir = ROOT / "data" / ds_name
        if not data_dir.exists():
            continue

        cat_cols = cfg["cat_cols"]
        target   = cfg["target"]
        keep     = cat_cols + [target]
        D = len(cat_cols)
        orders = list(range(1, D + 1))   # 1, 2, ..., D (full)

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

        print(f"\n=== {ds_name} (D={D}) ===")
        for k in orders:
            label = f"k={k}" + (" (full)" if k == D else "")
            print(f"  {label}", end="", flush=True)
            result = train_eval(train_df, val_df, test_df, cat_cols, target, k)
            result["dataset"] = ds_name
            rows.append(result)
            print(f"  |S|={result['num_subsets']}  RMSE={result['RMSE']:.4f}"
                  f"  NLL={result['NLL']:.4f}  Cov95={result['Cov95']:.3f}")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "ablation_kway.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
