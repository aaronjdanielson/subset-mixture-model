"""
Multi-seed robustness evaluation for SMM.

Runs SMM and all baselines across 5 random seeds on each dataset.
Seeds affect the train/val/test split (non-NBA datasets) and the
SMM weight initialisation. NBA always uses the fixed temporal split,
so only SMM initialisation varies.

Run from smm_paper/ root:
    python experiments/run_multiseed.py
"""

import sys, pathlib, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, BayesianRidge
from torch.utils.data import DataLoader
from scipy.stats import norm

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smm import (SubsetMaker, SubsetWeightsModel, SubsetDataset,
                 subset_mixture_neg_log_posterior, SubsetMixturePredictor,
                 compute_posterior_covariance, predict_with_uncertainty, coverage)
from experiments.run_experiments import (
    DATASETS, HP, rmse, mae, gaussian_nll, coverage_95,
    run_global_mean, run_linear, run_bayesian_ridge,
    run_lgbm, run_ngboost, run_mapie_lgbm
)

SEEDS = [42, 7, 13, 99, 2024]


def encode_and_split(df_full, cat_cols, target, seed, val_frac=0.15, test_frac=0.15):
    """Random split + encode. Returns (train, val, test) DataFrames."""
    train_val, test = train_test_split(df_full, test_size=test_frac, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_frac / (1 - test_frac), random_state=seed)
    for split in [train, val, test]:
        split.reset_index(drop=True, inplace=True)

    # Encode using training categories
    for col in cat_cols:
        if not pd.api.types.is_numeric_dtype(train[col]):
            cats = pd.CategoricalDtype(
                categories=train[col].astype("category").cat.categories)
            train[col] = train[col].astype(cats).cat.codes.astype(int)
            val[col]   = val[col].astype(cats).cat.codes.astype(int)
            test[col]  = test[col].astype(cats).cat.codes.astype(int)
    return train, val, test


def train_smm_seed(train_df, val_df, cat_cols, target, seed):
    torch.manual_seed(seed)
    subset_maker = SubsetMaker(train_df, cat_cols, [target])
    powerset = list(subset_maker.lookup.keys())
    model = SubsetWeightsModel(len(powerset))
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
    pi_hat = F.softmax(model.eta.detach(), dim=0)
    predictor = SubsetMixturePredictor(subset_maker, pi_hat)
    sigma_pi = compute_posterior_covariance(
        model, subset_maker, train_df, cat_cols, target, alpha=HP["alpha"])
    y_mean, y_std = predict_with_uncertainty(predictor, sigma_pi, val_df.assign(
        **{target: 0}))  # dummy call to warm up; then do test
    y_te = val_df[target].values  # placeholder — overwritten below
    return predictor, sigma_pi, model, subset_maker


def eval_smm_seed(predictor, sigma_pi, test_df, target):
    y_te = test_df[target].values
    y_mean, y_std = predict_with_uncertainty(predictor, sigma_pi, test_df)
    cov95 = coverage(y_te, y_mean, y_std, level=0.95)
    return {
        "RMSE": rmse(y_te, y_mean),
        "MAE":  mae(y_te, y_mean),
        "NLL":  gaussian_nll(y_te, y_mean, y_std),
        "Cov95": cov95,
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
        is_nba   = (ds_name == "nba")

        print(f"\n=== {ds_name} ===")

        # Load full dataset for re-splitting (non-NBA)
        if not is_nba:
            full_df = pd.concat([
                pd.read_csv(data_dir / "train.csv")[keep],
                pd.read_csv(data_dir / cfg["val_file"])[keep],
                pd.read_csv(data_dir / "test.csv")[keep],
            ], ignore_index=True).dropna()
            # Undo integer encoding for re-splitting (already encoded from download)
            # We just re-split the already-encoded data across seeds

        for seed in SEEDS:
            print(f"  seed={seed}", end="", flush=True)

            if is_nba:
                # Temporal split is fixed; only SMM init varies
                tr = pd.read_csv(data_dir / "train.csv")[keep].dropna().copy()
                va = pd.read_csv(data_dir / cfg["val_file"])[keep].dropna().copy()
                te = pd.read_csv(data_dir / "test.csv")[keep].dropna().copy()
                # Encode using training-set categories (mirrors run_experiments.py)
                for col in cat_cols:
                    if not pd.api.types.is_numeric_dtype(tr[col]):
                        cats = pd.CategoricalDtype(
                            categories=tr[col].astype("category").cat.categories)
                        tr[col] = tr[col].astype(cats).cat.codes.astype(int)
                        va[col] = va[col].astype(cats).cat.codes.astype(int)
                        te[col] = te[col].astype(cats).cat.codes.astype(int)
                train_df, val_df, test_df = tr, va, te
            else:
                # Re-split by seed
                train_df, val_df, test_df = encode_and_split(
                    full_df.copy(), cat_cols, target, seed)

            def row(method, metrics):
                return {"dataset": ds_name, "seed": seed,
                        "method": method, **metrics}

            # Baselines (deterministic given split, only vary across seeds via data)
            all_rows_seed = [
                row("GlobalMean",     run_global_mean(train_df, test_df, target)),
                row("Linear",         run_linear(train_df, test_df, cat_cols, target)),
                row("BayesianRidge",  run_bayesian_ridge(train_df, test_df, cat_cols, target)),
                row("LightGBM",       run_lgbm(train_df, test_df, cat_cols, target)),
                row("NGBoost",        run_ngboost(train_df, test_df, cat_cols, target)),
                row("MAPIE-LightGBM", run_mapie_lgbm(train_df, val_df, test_df,
                                                       cat_cols, target)),
            ]

            # SMM
            predictor, sigma_pi, model, subset_maker = train_smm_seed(
                train_df, val_df, cat_cols, target, seed)
            smm_metrics = eval_smm_seed(predictor, sigma_pi, test_df, target)
            all_rows_seed.append(row("SMM", smm_metrics))
            rows.extend(all_rows_seed)

            print(f"  SMM RMSE={smm_metrics['RMSE']:.4f} Cov={smm_metrics['Cov95']:.3f}")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "multiseed_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Summary: mean ± std per dataset × method
    summary = df.groupby(["dataset", "method"])[["RMSE", "MAE", "NLL", "Cov95"]].agg(
        ["mean", "std"]).round(4)
    print("\n\n=== Multi-seed summary (mean ± std) ===")
    print(summary.to_string())
    summary.to_csv(ROOT / "results" / "tables" / "multiseed_summary.csv")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
