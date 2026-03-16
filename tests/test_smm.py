"""
Core tests for the Subset Mixture Model package.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smm import (
    SubsetMaker,
    SubsetWeightsModel,
    SubsetDataset,
    subset_mixture_neg_log_posterior,
    SubsetMixturePredictor,
    compute_posterior_covariance,
    predict_with_uncertainty,
    coverage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """3-feature synthetic dataset with a known 3-way interaction."""
    rng = np.random.default_rng(0)
    n = 300
    a = rng.integers(0, 4, n)
    b = rng.integers(0, 3, n)
    c = rng.integers(0, 2, n)
    # target = interaction mean + noise
    cell_means = {(i, j, k): rng.normal(0, 2)
                  for i in range(4) for j in range(3) for k in range(2)}
    y = np.array([cell_means[(a[i], b[i], c[i])] + rng.normal(0, 0.1)
                  for i in range(n)])
    return pd.DataFrame({"a": a, "b": b, "c": c, "y": y})


@pytest.fixture
def splits(synthetic_df):
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(synthetic_df, test_size=0.2, random_state=0)
    tr, va = train_test_split(tr, test_size=0.15, random_state=0)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


CAT_COLS = ["a", "b", "c"]
TARGET   = "y"


# ---------------------------------------------------------------------------
# SubsetMaker
# ---------------------------------------------------------------------------

def test_subset_maker_powerset_size(splits):
    tr, _, _ = splits
    sm = SubsetMaker(tr, CAT_COLS, [TARGET])
    # 3 features → 2^3 - 1 = 7 subsets
    assert len(sm.lookup) == 7


def test_subset_maker_batch_lookup_shapes(splits):
    tr, _, _ = splits
    sm = SubsetMaker(tr, CAT_COLS, [TARGET])
    x = torch.tensor(tr[CAT_COLS].astype(np.float32).values[:16])
    mus, variances, mask = sm.batch_lookup(x)
    assert mus.shape == (16, 7)
    assert variances.shape == (16, 7)
    assert mask.shape == (16, 7)
    assert mask.dtype == torch.bool


def test_subset_maker_fallback(splits):
    tr, _, _ = splits
    sm = SubsetMaker(tr, CAT_COLS, [TARGET])
    assert np.isfinite(sm.fallback_mean)
    assert np.isfinite(sm.fallback_var)


# ---------------------------------------------------------------------------
# SubsetWeightsModel
# ---------------------------------------------------------------------------

def test_model_forward_returns_logits():
    model = SubsetWeightsModel(7)
    eta = model()
    assert eta.shape == (7,)
    assert not torch.isnan(eta).any()


def test_softmax_sums_to_one():
    model = SubsetWeightsModel(7)
    pi = F.softmax(model(), dim=0)
    assert torch.allclose(pi.sum(), torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def test_loss_finite(splits):
    tr, _, _ = splits
    sm = SubsetMaker(tr, CAT_COLS, [TARGET])
    model = SubsetWeightsModel(len(sm.lookup))
    loader = DataLoader(SubsetDataset(tr, CAT_COLS, [TARGET]),
                        batch_size=32, shuffle=False)
    x, y = next(iter(loader))
    mus, variances, mask = sm.batch_lookup(x)
    loss = subset_mixture_neg_log_posterior(model(), y, mus, variances, mask)
    assert torch.isfinite(loss)


def test_loss_decreases_after_training(splits):
    tr, va, _ = splits
    sm = SubsetMaker(tr, CAT_COLS, [TARGET])
    model = SubsetWeightsModel(len(sm.lookup))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = DataLoader(SubsetDataset(tr, CAT_COLS, [TARGET]),
                        batch_size=64, shuffle=True)

    def epoch_loss():
        total = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                mus, variances, mask = sm.batch_lookup(x)
                total += subset_mixture_neg_log_posterior(
                    model(), y, mus, variances, mask).item()
        return total

    loss_before = epoch_loss()
    model.train()
    for _ in range(5):
        for x, y in loader:
            opt.zero_grad()
            mus, variances, mask = sm.batch_lookup(x)
            subset_mixture_neg_log_posterior(
                model(), y, mus, variances, mask).backward()
            opt.step()
    loss_after = epoch_loss()
    assert loss_after < loss_before


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def test_predictor_output_shape(splits):
    tr, _, te = splits
    sm = SubsetMaker(tr, CAT_COLS, [TARGET])
    model = SubsetWeightsModel(len(sm.lookup))
    pi = F.softmax(model.eta.detach(), dim=0)
    predictor = SubsetMixturePredictor(sm, pi)
    preds = predictor.predict(te)
    assert preds.shape == (len(te),)
    assert np.isfinite(preds).all()


def test_predictor_fallback_for_unseen_cells():
    """Test point with values not seen in training should fall back gracefully."""
    rng = np.random.default_rng(1)
    tr = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 1, 1], "y": [1.0, 2.0, 3.0, 4.0]})
    sm = SubsetMaker(tr, ["a", "b"], ["y"])
    model = SubsetWeightsModel(len(sm.lookup))
    pi = F.softmax(model.eta.detach(), dim=0)
    predictor = SubsetMixturePredictor(sm, pi)
    # value 99 was never seen — should return fallback_mean
    unseen = pd.DataFrame({"a": [99], "b": [99], "y": [0.0]})
    preds = predictor.predict(unseen)
    assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------
# Laplace / uncertainty
# ---------------------------------------------------------------------------

def test_predict_with_uncertainty_shapes(splits):
    tr, va, te = splits
    sm = SubsetMaker(tr, CAT_COLS, [TARGET])
    model = SubsetWeightsModel(len(sm.lookup))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = DataLoader(SubsetDataset(tr, CAT_COLS, [TARGET]),
                        batch_size=64, shuffle=True)
    for _ in range(3):
        for x, y in loader:
            opt.zero_grad()
            mus, variances, mask = sm.batch_lookup(x)
            subset_mixture_neg_log_posterior(
                model(), y, mus, variances, mask).backward()
            opt.step()
    pi = F.softmax(model.eta.detach(), dim=0)
    predictor = SubsetMixturePredictor(sm, pi)
    sigma_pi = compute_posterior_covariance(
        model, sm, tr, CAT_COLS, TARGET, alpha=1.1)
    y_mean, y_std = predict_with_uncertainty(predictor, sigma_pi, te)
    assert y_mean.shape == (len(te),)
    assert y_std.shape == (len(te),)
    assert (y_std >= 0).all()


def test_coverage_function():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_mean = np.array([0.0, 1.0, 2.0, 3.0])
    y_std  = np.array([1.0, 1.0, 1.0, 1.0])
    cov = coverage(y_true, y_mean, y_std, level=0.95)
    assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# Synthetic recovery (integration test)
# ---------------------------------------------------------------------------

def test_synthetic_weight_recovery(splits):
    """
    With the DGP driven purely by the 3-way interaction,
    the full-powerset subset should receive the highest weight.
    """
    tr, va, te = splits
    sm = SubsetMaker(tr, CAT_COLS, [TARGET])
    model = SubsetWeightsModel(len(sm.lookup))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loader = DataLoader(SubsetDataset(tr, CAT_COLS, [TARGET]),
                        batch_size=64, shuffle=True)
    best_state, best_val = None, float("inf")
    val_loader = DataLoader(SubsetDataset(va, CAT_COLS, [TARGET]),
                            batch_size=64, shuffle=False)
    torch.manual_seed(0)
    for epoch in range(60):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            mus, variances, mask = sm.batch_lookup(x)
            subset_mixture_neg_log_posterior(
                model(), y, mus, variances, mask, alpha=1.1).backward()
            opt.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                mus, variances, mask = sm.batch_lookup(x)
                vl += subset_mixture_neg_log_posterior(
                    model(), y, mus, variances, mask, alpha=1.1).item()
        vl /= len(val_loader)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pi = F.softmax(model.eta.detach(), dim=0).numpy()
    subsets = list(sm.lookup.keys())
    # Full 3-way subset should be highest weight
    full_subset = tuple(sorted(CAT_COLS))
    idx = next(i for i, s in enumerate(subsets) if tuple(sorted(s)) == full_subset)
    assert pi[idx] == pi.max(), (
        f"3-way subset not top-weighted: pi={pi[idx]:.3f} vs max={pi.max():.3f}")
