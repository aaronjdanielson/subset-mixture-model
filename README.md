# Subset Mixture Model (SMM)

[![PyPI version](https://badge.fury.io/py/subset-mixture-model.svg)](https://pypi.org/project/subset-mixture-model/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SMM** is an interpretable, uncertainty-aware regression method for data with
categorical features. It learns a global convex mixture over *subset-induced
partition estimators*—one per non-empty feature subset—and returns a full
predictive distribution together with a transparent account of *why* each
prediction has the value and the uncertainty it does.

Version 0.2 upgrades the method to be genuinely probabilistic:

- **Cross-fitted weights** — subset cell means are multiway target encodings, so
  the mixture weights are learned on out-of-fold statistics to avoid target
  leakage (small, high-order cells no longer get to memorize the response).
- **Conjugate Student-t cells** — each cell uses a Normal-Inverse-Gamma
  posterior, so its predictive component is a Student-t that widens and grows
  heavy tails in sparse cells (the plug-in Gaussian is the large-sample limit).
- **Exact mixture inference** — NLL, CDF, central intervals (by bisection) and
  CRPS are computed exactly from the mixture of Student-t components.
- **Four-part predictive variance** — within-cell noise, cell-estimation
  uncertainty, subset-resolution disagreement, and weight uncertainty (Laplace).
- **Interpretation as diagnostics** — the learned weights are summarized by
  entropy, effective number of subsets, order-level mass, and concentration,
  rather than claimed to be a sparse ANOVA decomposition.

---

## Installation

```bash
pip install subset-mixture-model
```

Import as `smm`:

```python
import smm
```

Requires `torch>=2.0`, `numpy`, `pandas`, `scipy`. `plot_calibration` also needs
`matplotlib`.

---

## Quickstart

```python
import numpy as np
import pandas as pd
from smm import SMM

# --- synthetic data: a main effect (region) + an interaction (season x tier) ---
rng = np.random.default_rng(0)
N = 2000
region = rng.integers(0, 2, N)
season = rng.integers(0, 3, N)
tier   = rng.integers(0, 2, N)
y = 5.0 * region + 3.0 * (season == 1) * tier + rng.normal(0, 2.0, N)
df = pd.DataFrame({"region": region, "season": season, "tier": tier, "y": y})

train, val, test = df[:1400], df[1400:1700], df[1700:]
FEATURES, TARGET = ["region", "season", "tier"], "y"

# --- fit: cross-fitted weights + conjugate Student-t cells + Laplace UQ ---
model = SMM(FEATURES, TARGET, kappa0=1.0, lam=0.5).fit(train, val)

# --- point prediction and a full predictive distribution ---
mean = model.predict(test)
mean, std = model.predict_with_uncertainty(test)
lo, hi = model.interval(test, level=0.95)          # exact mixture interval

y_test = test[TARGET].values
print("RMSE :", np.sqrt(np.mean((mean - y_test) ** 2)).round(3))
print("NLL  :", round(model.nll(test, y_test), 3))
print("CRPS :", round(model.crps(test, y_test), 3))
print("cov95:", round(float(((y_test >= lo) & (y_test <= hi)).mean()), 3))
```

### Why this prediction? Uncertainty decomposition

```python
mean, std, aleatoric_std, epistemic_std = model.predict_with_uncertainty(
    test, return_components=True
)
# std**2 == aleatoric**2 + epistemic**2  (within-cell noise vs. everything else)
```

### Which interactions matter? Global diagnostics

```python
print(model.weight_table(top_k=5))     # subsets ranked by learned weight
print(model.diagnostics())             # H, N_eff, HHI, order-level mass M_k
```

`diagnostics()` returns the concentration of the weight distribution. Diffuse
weights (large `N_eff`) are an honest signal that *no* sparse subset explanation
dominates—prediction draws on many comparable resolutions—while concentrated
weights identify a few dominant interactions.

### Why *this* prediction? Local contributions

```python
row = test.iloc[[0]]
print(model.explain(row))     # per-subset contributions; they sum to the prediction
```

---

## What the model gives you

| Method | Returns |
|---|---|
| `SMM(features, target, ...)` | estimator; key knobs `kappa0` (shrinkage), `alpha0`,`lam` (order-aware prior), `K` (folds), `mode` (`"nig"`/`"plugin"`), `max_order` |
| `.fit(df, val_df=None)` | cross-fit → optimize weights (early-stop on `val_df`) → Laplace |
| `.predict(df)` | predictive mean |
| `.predict_with_uncertainty(df, return_components=)` | mean, std [, aleatoric, epistemic] |
| `.nll / .crps` | exact mixture proper scores |
| `.interval(df, level)` | exact central interval via bisection on the mixture CDF |
| `.weight_table / .diagnostics / .explain` | interpretability outputs |

Lower-level building blocks are also exported: `SubsetMaker`, `crossfit_components`,
`SubsetMixturePredictor`, `compute_posterior_covariance`, `predict_with_uncertainty`,
`NIGPrior`, `weight_table`, `weight_diagnostics`, `calibration_stats`.

To reproduce the first-version plug-in behavior, use
`SMM(..., mode="plugin", cross_fit=False, kappa0=0)`.

---

## Method summary

For each non-empty feature subset *s*, SMM groups the training data by the values
of *s* and models each resulting cell with a conjugate Normal-Inverse-Gamma
posterior, giving a Student-t predictive component. A single global simplex
weight vector π over all subsets is learned by MAP under an (optionally
order-aware) Dirichlet prior, using out-of-fold cell statistics. The predictive
distribution is the mixture ∑ₛ πₛ · tₛ, and uncertainty in π is propagated by a
Laplace approximation in the low-dimensional logit space.

SMM is intended for problems whose predictive structure is concentrated in a
modest number of naturally categorical features (full powerset for *D ≤ 8*;
order-restricted for larger *D*). It is a transparent, uncertainty-aware
alternative to gradient-boosted trees in that regime, not a general replacement.

---

## Citation

```bibtex
@article{danielson2026smm,
  title   = {Subset Mixture Models: Interpretable Probabilistic Aggregation
             of Partition Estimators for Categorical Regression},
  author  = {Danielson, Aaron John},
  journal = {Under review},
  year    = {2026},
}
```

## License

MIT © Aaron John Danielson
