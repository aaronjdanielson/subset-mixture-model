"""
Synthetic recovery experiment for SMM.

Generates data from a known DGP driven by a specific 3-way interaction,
then checks whether SMM recovers the correct subset weights.

Also runs a noise-feature robustness experiment: adds k=3 irrelevant
random categorical features to the NBA dataset and evaluates degradation.

Run from smm_paper/ root:
    python experiments/synthetic_recovery.py
"""

import sys, pathlib, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import OneHotEncoder

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from smm import (SubsetMaker, SubsetWeightsModel, SubsetDataset,
                 subset_mixture_neg_log_posterior, SubsetMixturePredictor,
                 compute_posterior_covariance, predict_with_uncertainty, coverage)
from experiments.run_experiments import HP, rmse, mae, gaussian_nll, coverage_95

# ── helpers ──────────────────────────────────────────────────────────────────

def train_smm(train_df, val_df, cat_cols, target, seed=42):
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
    return predictor, sigma_pi, pi_hat, list(subset_maker.lookup.keys())


# ── 1. Synthetic recovery ─────────────────────────────────────────────────────

def run_synthetic_recovery():
    print("\n=== Synthetic Recovery Experiment ===")
    rng = np.random.default_rng(42)

    # DGP: y is driven by a 3-way interaction (team × arena × weekday) + noise
    teams   = [f"T{i}" for i in range(6)]
    arenas  = [f"A{i}" for i in range(4)]
    weekdays = [f"D{i}" for i in range(7)]

    # Ground-truth 3-way means (random but fixed)
    truth = {}
    for t in teams:
        for a in arenas:
            for d in weekdays:
                truth[(t, a, d)] = rng.normal(0, 2)

    def gen(n):
        rows = []
        for _ in range(n):
            t = rng.choice(teams)
            a = rng.choice(arenas)
            d = rng.choice(weekdays)
            y = truth[(t, a, d)] + rng.normal(0, 0.3)
            rows.append({"team": t, "arena": a, "weekday": d, "y": y})
        return pd.DataFrame(rows)

    train_df = gen(2000)
    val_df   = gen(400)
    test_df  = gen(400)

    cat_cols = ["team", "arena", "weekday"]
    target   = "y"

    # Encode string categories to integers (required by SubsetDataset)
    for col in cat_cols:
        cats = sorted(set(train_df[col]))
        cat_map = {c: i for i, c in enumerate(cats)}
        for df_ in [train_df, val_df, test_df]:
            df_[col] = df_[col].map(cat_map)

    predictor, sigma_pi, pi_hat, subsets = train_smm(
        train_df, val_df, cat_cols, target)

    # Identify which subset is ("team", "arena", "weekday")
    # Subsets are stored as tuples
    target_subset = tuple(sorted(cat_cols))
    subset_strs = [str(tuple(sorted(s))) if isinstance(s, list) else str(tuple(sorted(s)))
                   for s in subsets]

    pi_arr = pi_hat.numpy()
    sorted_idx = np.argsort(pi_arr)[::-1]

    print("\nTop-10 learned weights:")
    for rank, idx in enumerate(sorted_idx[:10]):
        s = subsets[idx]
        flag = " ← TRUE INTERACTION" if tuple(sorted(s)) == tuple(sorted(cat_cols)) else ""
        print(f"  {rank+1:2d}. {s}  π={pi_arr[idx]:.4f}{flag}")

    # Find rank of the true 3-way subset
    true_subset = tuple(sorted(cat_cols))
    ranks = []
    for idx in range(len(subsets)):
        if tuple(sorted(subsets[idx])) == true_subset:
            ranks.append(list(sorted_idx).index(idx) + 1)
    true_rank = ranks[0] if ranks else -1
    true_weight = pi_arr[[i for i, s in enumerate(subsets)
                           if tuple(sorted(s)) == true_subset][0]]

    print(f"\nTrue 3-way interaction rank: {true_rank} / {len(subsets)}")
    print(f"True 3-way interaction weight: {true_weight:.4f}")
    print(f"Expected for uniform: {1/len(subsets):.4f}")

    # Evaluate on test set
    y_mean, y_std = predict_with_uncertainty(predictor, sigma_pi, test_df)
    y_te = test_df[target].values
    print(f"\nTest RMSE: {rmse(y_te, y_mean):.4f}")
    print(f"Test Cov95: {coverage(y_te, y_mean, y_std, level=0.95):.3f}")

    return {
        "true_rank": true_rank,
        "true_weight": float(true_weight),
        "n_subsets": len(subsets),
        "test_rmse": rmse(y_te, y_mean),
        "test_cov95": float(coverage(y_te, y_mean, y_std, level=0.95)),
    }


# ── 2. Noise-feature robustness ───────────────────────────────────────────────

def run_noise_robustness():
    """Use Ames Housing (D=5, n~1460) for speed; NBA would require D→9 = 511 subsets."""
    print("\n=== Noise-Feature Robustness Experiment (Ames Housing) ===")
    data_dir = ROOT / "data" / "ames_housing"
    if not data_dir.exists():
        print("Ames Housing data not found, skipping.")
        return

    cat_cols = ["Neighborhood", "BldgType", "HouseStyle", "Foundation", "GarageType"]
    target = "log_price"
    keep_cols = cat_cols + [target]

    tr = pd.read_csv(data_dir / "train.csv")[keep_cols].dropna().copy()
    va = pd.read_csv(data_dir / "val.csv")[keep_cols].dropna().copy()
    te = pd.read_csv(data_dir / "test.csv")[keep_cols].dropna().copy()

    # Encode string categories to integers
    for col in cat_cols:
        if not pd.api.types.is_numeric_dtype(tr[col]):
            cats = pd.CategoricalDtype(
                categories=tr[col].astype("category").cat.categories)
            tr[col] = tr[col].astype(cats).cat.codes.astype(int)
            va[col] = va[col].astype(cats).cat.codes.astype(int)
            te[col] = te[col].astype(cats).cat.codes.astype(int)

    # No noise features (baseline)
    pred0, sig0, _, _ = train_smm(tr, va, cat_cols, target)
    y_te = te[target].values
    y_mean0, _ = predict_with_uncertainty(pred0, sig0, te)
    rmse0 = rmse(y_te, y_mean0)
    print(f"\nBaseline (D=5, |S|=31): RMSE={rmse0:.4f}")

    rng = np.random.default_rng(0)
    results = [{"n_noise": 0, "D": 5, "S": 31, "RMSE": rmse0, "noise_weight": float("nan")}]

    for n_noise in [1, 2, 3]:
        tr2, va2, te2 = tr.copy(), va.copy(), te.copy()
        noise_cols = []
        for k in range(n_noise):
            col = f"noise_{k}"
            tr2[col] = rng.integers(0, 10, len(tr2))
            va2[col] = rng.integers(0, 10, len(va2))
            te2[col] = rng.integers(0, 10, len(te2))
            noise_cols.append(col)
        cols = cat_cols + noise_cols
        pred_n, sig_n, pi_n, subsets_n = train_smm(tr2, va2, cols, target)
        y_mean_n, _ = predict_with_uncertainty(pred_n, sig_n, te2)
        r = rmse(y_te, y_mean_n)
        noise_weight = sum(
            float(pi_n[i]) for i, s in enumerate(subsets_n)
            if all(f in noise_cols for f in s)
        )
        print(f"  +{n_noise} noise (D={5+n_noise}, |S|={len(subsets_n)}): "
              f"RMSE={r:.4f}  noise-only weight={noise_weight:.4f}")
        results.append({"n_noise": n_noise, "D": 5+n_noise, "S": len(subsets_n),
                        "RMSE": r, "noise_weight": noise_weight})

    return results


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rec = run_synthetic_recovery()
    noise = run_noise_robustness()

    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([rec]).to_csv(out_dir / "synthetic_recovery.csv", index=False)
    if noise:
        pd.DataFrame(noise).to_csv(out_dir / "noise_robustness.csv", index=False)
    print("\nDone.")
