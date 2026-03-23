"""
Run SMM and baselines on all datasets, write results to results/tables/.

Usage (from smm_paper/ root):
    python experiments/run_experiments.py

Outputs
-------
results/tables/results_summary.csv   — RMSE / MAE / NLL / Coverage per model×dataset
results/models/<dataset>_predictor.joblib  — saved SMM predictor per dataset
results/figures/<dataset>_weights.png      — top-subset weight bar chart
"""

import sys
import pathlib
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smm import SubsetMaker, SubsetWeightsModel, SubsetDataset
from smm import subset_mixture_neg_log_posterior, SubsetMixturePredictor
from smm import compute_posterior_covariance, predict_with_uncertainty, coverage

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------
DATASETS = {
    "nba": {
        "cat_cols": ["home_team", "visitor_team", "arena_name",
                     "start_hour", "day_of_week", "month"],
        "target": "attendance",
        "val_file": "val.csv",
    },
    "bike_sharing": {
        "cat_cols": ["season", "holiday", "weekday", "weathersit", "workingday"],
        "target": "log_cnt",
        "val_file": "val.csv",
    },
    "student_performance": {
        "cat_cols": ["school", "sex", "address", "Mjob", "Fjob"],
        "target": "G3",
        "val_file": "val.csv",
    },
    "ames_housing": {
        "cat_cols": ["Neighborhood", "BldgType", "HouseStyle",
                     "Foundation", "GarageType"],
        "target": "log_price",
        "val_file": "val.csv",
    },
    "diamonds": {
        "cat_cols": ["cut", "color", "clarity"],
        "target": "log_price",
        "val_file": "val.csv",
    },
    "forest_fires": {
        "cat_cols": ["month", "day", "X", "Y"],
        "target": "log_area",
        "val_file": "val.csv",
    },
}

# Training hyperparameters
HP = dict(
    num_epochs=150,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-5,
    alpha=1.1,
    patience=10,
    min_epochs=10,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def gaussian_nll(y_true, y_mean, y_std):
    """Mean NLL under N(y_mean, y_std^2)."""
    return float(-norm.logpdf(y_true, loc=y_mean, scale=y_std).mean())


def coverage_95(y_true, y_mean, y_std):
    """Empirical coverage of the 95% CI [mean ± 1.96*std]."""
    lo = y_mean - 1.96 * y_std
    hi = y_mean + 1.96 * y_std
    return float(((y_true >= lo) & (y_true <= hi)).mean())


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_global_mean(train_df, test_df, target):
    mu = train_df[target].mean()
    std = train_df[target].std()
    y = test_df[target].values
    y_hat = np.full_like(y, mu)
    return {
        "RMSE": rmse(y, y_hat),
        "MAE":  mae(y, y_hat),
        "NLL":  gaussian_nll(y, y_hat, np.full_like(y, std)),
        "Cov95": coverage_95(y, y_hat, np.full_like(y, std)),
    }


def run_linear(train_df, test_df, cat_cols, target):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(train_df[cat_cols].astype(str))
    X_te = enc.transform(test_df[cat_cols].astype(str))
    y_tr = train_df[target].values
    y_te = test_df[target].values

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    resid_std = float(np.std(y_tr - model.predict(X_tr)))
    return {
        "RMSE": rmse(y_te, y_hat),
        "MAE":  mae(y_te, y_hat),
        "NLL":  gaussian_nll(y_te, y_hat, np.full_like(y_hat, resid_std)),
        "Cov95": coverage_95(y_te, y_hat, np.full_like(y_hat, resid_std)),
    }


def run_bayesian_ridge(train_df, test_df, cat_cols, target):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(train_df[cat_cols].astype(str))
    X_te = enc.transform(test_df[cat_cols].astype(str))
    y_tr = train_df[target].values
    y_te = test_df[target].values

    model = BayesianRidge()
    model.fit(X_tr, y_tr)
    y_hat, y_std = model.predict(X_te, return_std=True)
    return {
        "RMSE": rmse(y_te, y_hat),
        "MAE":  mae(y_te, y_hat),
        "NLL":  gaussian_nll(y_te, y_hat, y_std),
        "Cov95": coverage_95(y_te, y_hat, y_std),
    }


def run_ngboost(train_df, test_df, cat_cols, target):
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
    except ImportError:
        print("  NGBoost not installed — skipping.")
        return {"RMSE": None, "MAE": None, "NLL": None, "Cov95": None}

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(train_df[cat_cols].astype(str))
    X_te = enc.transform(test_df[cat_cols].astype(str))
    y_tr = train_df[target].values
    y_te = test_df[target].values

    model = NGBRegressor(Dist=Normal, n_estimators=500, verbose=False,
                         random_state=42)
    model.fit(X_tr, y_tr)
    pred_dist = model.pred_dist(X_te)
    y_hat = pred_dist.loc
    y_std = pred_dist.scale
    return {
        "RMSE":  rmse(y_te, y_hat),
        "MAE":   mae(y_te, y_hat),
        "NLL":   gaussian_nll(y_te, y_hat, y_std),
        "Cov95": coverage_95(y_te, y_hat, y_std),
    }


def run_mapie_lgbm(train_df, val_df, test_df, cat_cols, target):
    """LightGBM wrapped with split-conformal MAPIE for calibrated coverage."""
    try:
        import lightgbm as lgb
        from mapie.regression import MapieRegressor
    except ImportError:
        print("  MAPIE or LightGBM not installed — skipping.")
        return {"RMSE": None, "MAE": None, "NLL": None, "Cov95": None}

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(train_df[cat_cols].astype(str))
    X_va = enc.transform(val_df[cat_cols].astype(str))
    X_te = enc.transform(test_df[cat_cols].astype(str))
    y_tr = train_df[target].values
    y_va = val_df[target].values
    y_te = test_df[target].values

    # Combine train+val for calibration: fit on train, calibrate on val
    base = lgb.LGBMRegressor(n_estimators=300, num_leaves=31,
                              random_state=42, verbose=-1)
    base.fit(X_tr, y_tr)

    mapie = MapieRegressor(estimator=base, method="base", cv="prefit")
    mapie.fit(X_va, y_va)

    y_hat, pi = mapie.predict(X_te, alpha=0.05)   # 95% PI
    lower = pi[:, 0, 0]
    upper = pi[:, 1, 0]
    # Infer a per-point sigma from the half-width (for NLL)
    y_std = np.maximum((upper - lower) / (2 * 1.96), 1e-6)
    cov95 = float(((y_te >= lower) & (y_te <= upper)).mean())
    return {
        "RMSE":  rmse(y_te, y_hat),
        "MAE":   mae(y_te, y_hat),
        "NLL":   gaussian_nll(y_te, y_hat, y_std),
        "Cov95": cov95,
    }


def run_lgbm(train_df, test_df, cat_cols, target):
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not installed — skipping.")
        return {"RMSE": None, "MAE": None, "NLL": None, "Cov95": None}

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(train_df[cat_cols].astype(str))
    X_te = enc.transform(test_df[cat_cols].astype(str))
    y_tr = train_df[target].values
    y_te = test_df[target].values

    model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, random_state=42,
                               verbose=-1)
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    resid_std = float(np.std(y_tr - model.predict(X_tr)))
    return {
        "RMSE": rmse(y_te, y_hat),
        "MAE":  mae(y_te, y_hat),
        "NLL":  gaussian_nll(y_te, y_hat, np.full_like(y_hat, resid_std)),
        "Cov95": coverage_95(y_te, y_hat, np.full_like(y_hat, resid_std)),
    }


def run_catboost(train_df, test_df, cat_cols, target):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("  CatBoost not installed — skipping.")
        return {"RMSE": None, "MAE": None, "NLL": None, "Cov95": None}

    # CatBoost accepts integer-encoded categoricals natively via cat_features.
    X_tr = train_df[cat_cols].values
    X_te = test_df[cat_cols].values
    y_tr = train_df[target].values
    y_te = test_df[target].values

    cat_idx = list(range(len(cat_cols)))
    model = CatBoostRegressor(iterations=300, random_seed=42, verbose=0)
    model.fit(X_tr, y_tr, cat_features=cat_idx)
    y_hat = model.predict(X_te)
    resid_std = float(np.std(y_tr - model.predict(X_tr)))
    return {
        "RMSE":  rmse(y_te, y_hat),
        "MAE":   mae(y_te, y_hat),
        "NLL":   gaussian_nll(y_te, y_hat, np.full_like(y_hat, resid_std)),
        "Cov95": coverage_95(y_te, y_hat, np.full_like(y_hat, resid_std)),
    }


def run_mapie_catboost(train_df, val_df, test_df, cat_cols, target):
    """CatBoost wrapped with split-conformal MAPIE for calibrated coverage."""
    try:
        from catboost import CatBoostRegressor
        from mapie.regression import MapieRegressor
    except ImportError:
        print("  CatBoost or MAPIE not installed — skipping.")
        return {"RMSE": None, "MAE": None, "NLL": None, "Cov95": None}

    X_tr = train_df[cat_cols].values
    X_va = val_df[cat_cols].values
    X_te = test_df[cat_cols].values
    y_tr = train_df[target].values
    y_va = val_df[target].values
    y_te = test_df[target].values

    cat_idx = list(range(len(cat_cols)))
    base = CatBoostRegressor(iterations=300, random_seed=42, verbose=0)
    base.fit(X_tr, y_tr, cat_features=cat_idx)

    mapie = MapieRegressor(estimator=base, method="base", cv="prefit")
    mapie.fit(X_va, y_va)

    y_hat, pi = mapie.predict(X_te, alpha=0.05)
    lower = pi[:, 0, 0]
    upper = pi[:, 1, 0]
    y_std = np.maximum((upper - lower) / (2 * 1.96), 1e-6)
    cov95 = float(((y_te >= lower) & (y_te <= upper)).mean())
    return {
        "RMSE":  rmse(y_te, y_hat),
        "MAE":   mae(y_te, y_hat),
        "NLL":   gaussian_nll(y_te, y_hat, y_std),
        "Cov95": cov95,
    }


# ---------------------------------------------------------------------------
# SMM training
# ---------------------------------------------------------------------------

def train_smm(train_df, val_df, cat_cols, target, dataset_name):
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

    print(f"  Training SMM: {num_subsets} subsets, "
          f"{len(train_df)} train / {len(val_df)} val rows")

    for epoch in range(HP["num_epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pi_logits = model(x)
            mus, variances, mask = subset_maker.batch_lookup(x)
            loss = subset_mixture_neg_log_posterior(
                pi_logits, y, mus, variances, mask, alpha=HP["alpha"]
            )
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pi_logits = model(x)
                mus, variances, mask = subset_maker.batch_lookup(x)
                val_loss += subset_mixture_neg_log_posterior(
                    pi_logits, y, mus, variances, mask, alpha=HP["alpha"]
                ).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= HP["patience"] and epoch >= HP["min_epochs"]:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_weights = F.softmax(model.eta.detach().cpu(), dim=0)
    predictor = SubsetMixturePredictor(subset_maker, final_weights)

    # Save predictor
    models_dir = ROOT / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, models_dir / f"{dataset_name}_predictor.joblib")

    # Save weight chart
    _plot_weights(final_weights.numpy(), powerset, dataset_name)

    return predictor, subset_maker, model


def _plot_weights(weights, powerset, dataset_name):
    top_n = min(10, len(weights))
    top_idx = np.argsort(weights)[-top_n:][::-1]
    labels = ["+".join(list(powerset[i])) for i in top_idx]
    vals   = weights[top_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(top_n), vals[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Mixture weight")
    ax.set_title(f"Top subset weights — {dataset_name}")
    plt.tight_layout()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{dataset_name}_weights.png", dpi=150)
    plt.close(fig)


def eval_smm(predictor, model, subset_maker, train_df, test_df, cat_cols, target, alpha):
    y_te = test_df[target].values

    print("    Computing Laplace posterior covariance...")
    sigma_pi = compute_posterior_covariance(
        model, subset_maker, train_df, cat_cols, target, alpha=alpha
    )

    y_mean, y_std, aleatoric_std, epistemic_std = predict_with_uncertainty(
        predictor, sigma_pi, test_df, return_components=True
    )

    cov95 = coverage(y_te, y_mean, y_std, level=0.95)

    print(f"    Aleatoric std (mean): {aleatoric_std.mean():.4f} | "
          f"Epistemic std (mean): {epistemic_std.mean():.4f} | "
          f"Coverage 95%: {cov95:.3f}")

    return {
        "RMSE":  rmse(y_te, y_mean),
        "MAE":   mae(y_te, y_mean),
        "NLL":   gaussian_nll(y_te, y_mean, y_std),
        "Cov95": cov95,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    all_rows = []

    for ds_name, cfg in DATASETS.items():
        data_dir = ROOT / "data" / ds_name
        if not data_dir.exists():
            print(f"\nSkipping {ds_name}: data not found at {data_dir}")
            print("  Run: python data/download_datasets.py")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        cat_cols = cfg["cat_cols"]
        target   = cfg["target"]
        keep     = cat_cols + [target]

        train_df = pd.read_csv(data_dir / "train.csv")[keep].dropna().copy()
        val_df   = pd.read_csv(data_dir / cfg["val_file"])[keep].dropna().copy()
        test_df  = pd.read_csv(data_dir / "test.csv")[keep].dropna().copy()

        # Integer-encode any cat column that is not already numeric.
        # Encode column-by-column since some datasets are partially pre-encoded.
        for col in cat_cols:
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                cats = pd.CategoricalDtype(
                    categories=train_df[col].astype("category").cat.categories
                )
                train_df[col] = train_df[col].astype(cats).cat.codes.astype(int)
                val_df[col]   = val_df[col].astype(cats).cat.codes.astype(int)
                test_df[col]  = test_df[col].astype(cats).cat.codes.astype(int)

        def row(method, metrics):
            return {"dataset": ds_name, "method": method, **metrics}

        print("  Global Mean...")
        all_rows.append(row("GlobalMean",    run_global_mean(train_df, test_df, target)))
        print("  Linear Regression...")
        all_rows.append(row("Linear",        run_linear(train_df, test_df, cat_cols, target)))
        print("  Bayesian Ridge...")
        all_rows.append(row("BayesianRidge", run_bayesian_ridge(train_df, test_df, cat_cols, target)))
        print("  LightGBM...")
        all_rows.append(row("LightGBM",      run_lgbm(train_df, test_df, cat_cols, target)))
        print("  NGBoost...")
        all_rows.append(row("NGBoost",       run_ngboost(train_df, test_df, cat_cols, target)))
        print("  MAPIE-LightGBM (conformal)...")
        all_rows.append(row("MAPIE-LightGBM", run_mapie_lgbm(train_df, val_df, test_df, cat_cols, target)))
        print("  CatBoost...")
        all_rows.append(row("CatBoost",       run_catboost(train_df, test_df, cat_cols, target)))
        print("  MAPIE-CatBoost (conformal)...")
        all_rows.append(row("MAPIE-CatBoost", run_mapie_catboost(train_df, val_df, test_df, cat_cols, target)))

        print("  SMM...")
        predictor, subset_maker, model = train_smm(train_df, val_df, cat_cols, target, ds_name)
        all_rows.append(row("SMM", eval_smm(
            predictor, model, subset_maker, train_df, test_df,
            cat_cols, target, alpha=HP["alpha"]
        )))

    results_df = pd.DataFrame(all_rows)
    tables_dir = ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "results_summary.csv", index=False)

    print("\n\nResults summary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved to {tables_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
