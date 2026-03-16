# Subset Mixture Model (SMM)

**SMM** is an interpretable, empirical-Bayes method for regression on datasets with categorical features. It aggregates partition-based conditional-mean estimators over all non-empty feature subsets using learned simplex weights, adaptively balancing bias and variance across partition granularities.

## Key idea

Each feature subset $s$ induces a partition of the covariate space and a natural estimator of the conditional expectation — its empirical cell mean. SMM learns a convex combination of these estimators:

$$\hat{f}(\mathbf{x}) = \sum_{s \in \mathcal{S}} \hat{\pi}_s \cdot \hat{\mu}_{m(s,\mathbf{x})}(s)$$

The learned weights $\hat{\pi}_s$ are directly interpretable: they reveal which feature interactions drive predictions on average. Uncertainty is propagated from the MAP weight estimates via a Laplace approximation, yielding aleatoric/epistemic decompositions without post-hoc calibration.

## Installation

```bash
pip install subset-mixture-model
```

Or from source:

```bash
git clone https://github.com/aaronjdanielson/subset-mixture-model
cd subset-mixture-model
pip install -e .
```

## Quick start

```python
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smm import (
    SubsetMaker, SubsetWeightsModel, SubsetDataset,
    subset_mixture_neg_log_posterior, SubsetMixturePredictor,
    compute_posterior_covariance, predict_with_uncertainty,
)

# --- your data (integer-coded categorical features) ---
train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")

cat_cols = ["feature_a", "feature_b", "feature_c"]
target   = "y"

# --- build lookup table ---
subset_maker = SubsetMaker(train_df, cat_cols, [target])
n_subsets = len(subset_maker.lookup)

# --- train ---
model     = SubsetWeightsModel(n_subsets)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loader    = DataLoader(SubsetDataset(train_df, cat_cols, [target]),
                       batch_size=64, shuffle=True)

for epoch in range(100):
    for x, y in loader:
        optimizer.zero_grad()
        mus, variances, mask = subset_maker.batch_lookup(x)
        loss = subset_mixture_neg_log_posterior(
            model(), y, mus, variances, mask, alpha=1.1)
        loss.backward()
        optimizer.step()

# --- predict with uncertainty ---
pi_hat    = F.softmax(model.eta.detach(), dim=0)
predictor = SubsetMixturePredictor(subset_maker, pi_hat)
sigma_pi  = compute_posterior_covariance(
    model, subset_maker, train_df, cat_cols, target, alpha=1.1)

y_mean, y_std = predict_with_uncertainty(predictor, sigma_pi, test_df)
# y_mean: point predictions
# y_std:  total predictive standard deviation (aleatoric + epistemic)
```

## Interpretability

```python
import numpy as np

subsets = list(subset_maker.lookup.keys())
top_idx = np.argsort(pi_hat.numpy())[::-1][:10]

for rank, i in enumerate(top_idx):
    print(f"{rank+1:2d}. {subsets[i]}  π={pi_hat[i]:.4f}")
```

## Features

- **Interpretable by construction**: learned weights reveal which feature interactions matter
- **Principled uncertainty**: aleatoric/epistemic decomposition via Laplace approximation
- **Efficient training**: only $2^D - 1$ logits optimized; lookup table precomputed once
- **No post-hoc calibration**: well-calibrated predictive intervals out of the box
- **Scalable to D ≤ 15** features; $k$-way truncation available for larger $D$

## Datasets supported

Any tabular dataset with integer-coded (or string, with encoding) categorical features and a continuous target.

## Citation

```bibtex
@article{danielson2025smm,
  title   = {Subset Mixture Model: Interpretable Empirical-Bayes Aggregation
             of Partition Estimators for Categorical Regression},
  author  = {Danielson, Aaron John},
  journal = {Machine Learning},
  year    = {2025},
  note    = {Under review}
}
```

## License

MIT
