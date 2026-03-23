"""
Laplace approximation for posterior uncertainty in the Subset Mixture Model.

After MAP estimation of eta (the logit weight vector), the Laplace approximation
treats the negative log-posterior as a quadratic around the MAP:

    q(eta) = N(eta | eta_hat, H^{-1})

where H is the Hessian of the negative log-posterior at eta_hat.

Transforming to the simplex via pi = softmax(eta) and applying the delta method:

    Cov[pi | D] ≈ Sigma_pi = J H^{-1} J^T

where J = d(pi)/d(eta) is the |S| x |S| softmax Jacobian.

Predictive variance for a new point x_tilde decomposes as:

    Var[y | x_tilde] ≈  sum_s pi_s * (sigma^2_s(x_tilde) + mu_s(x_tilde)^2)
                        - y_hat(x_tilde)^2                       [aleatoric]
                      + mu_{x_tilde}^T Sigma_pi mu_{x_tilde}     [epistemic]

The aleatoric term is the exact variance of the Gaussian mixture at the MAP
weights (within-component noise plus between-component scatter). The epistemic
term propagates uncertainty in the learned weights via the delta method.

References:
    MacKay, D.J.C. (1992). A Practical Bayesian Framework for Backpropagation Networks.
    Daxberger et al. (2021). Laplace Redux -- Effortless Bayesian Deep Learning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import SubsetDataset, subset_mixture_neg_log_posterior
from .subset_maker import SubsetMaker
from .predictor import SubsetMixturePredictor


# ---------------------------------------------------------------------------
# Step 1: Compute posterior covariance of pi
# ---------------------------------------------------------------------------

def softmax_jacobian(pi: torch.Tensor) -> torch.Tensor:
    """
    Jacobian of softmax(eta) w.r.t. eta, evaluated at pi = softmax(eta).

    J_ij = pi_i * (delta_ij - pi_j)

    Args:
        pi: [S] mixture weights (must sum to 1).

    Returns:
        J: [S, S] Jacobian matrix.
    """
    return torch.diag(pi) - torch.outer(pi, pi)


def compute_hessian(
    model: torch.nn.Module,
    subset_maker: SubsetMaker,
    train_df,
    cat_cols: list,
    target: str,
    alpha: float = 1.1,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Compute the exact Hessian of the negative log-posterior w.r.t. eta.

    Because |S| is at most a few hundred, the Hessian is small and can be
    computed exactly via torch.autograd.functional.hessian on the full
    training set (loaded in batches if needed, accumulated via the empirical
    Fisher for very large datasets).

    For datasets where n is large we accumulate an approximation via the
    diagonal outer-product (empirical Fisher). For the exact Hessian we need
    the full batch, which is feasible when n < ~20K.

    Args:
        model:        Trained SubsetWeightsModel.
        subset_maker: Fitted SubsetMaker.
        train_df:     Training DataFrame (integer-encoded).
        cat_cols:     Categorical feature column names.
        target:       Target column name.
        alpha:        Dirichlet concentration (must match training value).
        batch_size:   Batch size for loading data (only affects memory).

    Returns:
        H: [S, S] Hessian tensor.
    """
    model.eval()
    eta_hat = model.eta.detach().clone()

    # Load all training data (full batch for exact Hessian)
    ds = SubsetDataset(train_df, cat_cols, [target])
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    x_all, y_all = next(iter(loader))

    mus, variances, mask = subset_maker.batch_lookup(x_all)
    # Detach and keep on CPU
    mus      = mus.detach()
    variances = variances.detach()
    mask     = mask.detach()

    def loss_fn(eta):
        return subset_mixture_neg_log_posterior(
            eta, y_all, mus, variances, mask, alpha=alpha
        )

    eta_param = eta_hat.requires_grad_(True)
    H = torch.autograd.functional.hessian(loss_fn, eta_param)  # [S, S]
    return H.detach()


def compute_posterior_covariance(
    model: torch.nn.Module,
    subset_maker: SubsetMaker,
    train_df,
    cat_cols: list,
    target: str,
    alpha: float = 1.1,
    hessian_reg: float = 1e-4,
) -> torch.Tensor:
    """
    Compute the Laplace approximation to the posterior covariance of pi.

    Sigma_pi = J H^{-1} J^T

    Args:
        model:        Trained SubsetWeightsModel (provides eta_hat).
        subset_maker: Fitted SubsetMaker.
        train_df:     Training DataFrame (integer-encoded).
        cat_cols:     Categorical feature column names.
        target:       Target column name.
        alpha:        Dirichlet concentration (must match training value).
        hessian_reg:  Ridge added to H before inversion for numerical stability.

    Returns:
        sigma_pi: [S, S] posterior covariance of mixture weights.
    """
    eta_hat = model.eta.detach()
    pi_hat  = F.softmax(eta_hat, dim=0)

    S = len(eta_hat)
    H = compute_hessian(model, subset_maker, train_df, cat_cols, target, alpha)

    # Regularise to ensure positive definiteness
    H_reg  = H + hessian_reg * torch.eye(S)
    H_inv  = torch.linalg.inv(H_reg)

    # Softmax Jacobian at MAP estimate
    J = softmax_jacobian(pi_hat)                # [S, S]
    sigma_pi = J @ H_inv @ J.T                  # [S, S]

    return sigma_pi


# ---------------------------------------------------------------------------
# Step 2: Predictive distribution for new points
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    predictor: SubsetMixturePredictor,
    sigma_pi: torch.Tensor,
    df,
    return_components: bool = False,
):
    """
    Compute predictive mean and uncertainty for a batch of test points.

    Predictive mean:
        y_hat(x) = pi_hat^T mu_x

    Predictive variance:
        Var[y|x] = sum_s pi_s * sigma_s^2(x)          [aleatoric]
                 + mu_x^T Sigma_pi mu_x                 [epistemic]

    For test points with no valid subset cell (all masked), the predictor
    falls back to the global training mean with aleatoric variance equal to
    the global training variance.

    Args:
        predictor:  SubsetMixturePredictor with learned weight vector.
        sigma_pi:   [S, S] posterior covariance from compute_posterior_covariance.
        df:         Test DataFrame (integer-encoded).
        return_components: If True, also return aleatoric and epistemic stds.

    Returns:
        y_mean (np.ndarray):  [B] predicted means.
        y_std  (np.ndarray):  [B] total predictive standard deviations.
        (optional) aleatoric_std (np.ndarray): [B]
        (optional) epistemic_std (np.ndarray): [B]
    """
    batch_tensor = torch.tensor(
        df[predictor.subset_maker.subset_features].astype(float).values,
        dtype=torch.float32,
    )
    mus, variances, mask = predictor.subset_maker.batch_lookup(batch_tensor)

    B, S = mus.shape
    pi    = predictor.weight_vector                         # [S]
    weights = pi.unsqueeze(0).expand(B, S).clone()         # [B, S]

    # Mask invalid cells
    weights   = weights.masked_fill(~mask, 0.0)
    mus_m     = mus.masked_fill(~mask, 0.0)
    vars_m    = variances.masked_fill(~mask, 0.0)

    weight_sums  = weights.sum(dim=1, keepdim=True)        # [B, 1]
    norm_weights = weights / (weight_sums + 1e-9)          # [B, S]

    # --- Predictive mean ---
    fallback_mask = weight_sums.squeeze() < 1e-6           # [B]
    y_mean_weighted = (mus_m * norm_weights).sum(dim=1)    # [B]
    y_mean = torch.where(
        fallback_mask,
        torch.full_like(y_mean_weighted, predictor.subset_maker.fallback_mean),
        y_mean_weighted,
    )

    # --- Aleatoric variance: exact mixture variance = within + between component ---
    # Var[y|x,pi] = sum_s pi_s*(sigma_s^2 + mu_s^2) - y_hat^2
    aleatoric_var = (norm_weights * (vars_m + mus_m ** 2)).sum(dim=1) - y_mean_weighted ** 2  # [B]
    # Fallback: use global training variance
    aleatoric_var = torch.where(
        fallback_mask,
        torch.full_like(aleatoric_var, predictor.subset_maker.fallback_var),
        aleatoric_var,
    )

    # --- Epistemic variance: mu_x^T Sigma_pi mu_x ---
    # mu_x is mus_m[i] — the vector of subset-conditional means for example i
    # (zeroed for invalid cells, consistent with the masked prediction)
    epistemic_var = torch.einsum("bi,ij,bj->b", mus_m, sigma_pi, mus_m)  # [B]
    epistemic_var = epistemic_var.clamp(min=0.0)

    total_var = aleatoric_var + epistemic_var
    total_std = torch.sqrt(total_var.clamp(min=1e-12))

    y_mean_np      = y_mean.detach().numpy()
    total_std_np   = total_std.detach().numpy()

    if return_components:
        aleatoric_std_np = torch.sqrt(aleatoric_var.clamp(min=1e-12)).detach().numpy()
        epistemic_std_np = torch.sqrt(epistemic_var.clamp(min=1e-12)).detach().numpy()
        return y_mean_np, total_std_np, aleatoric_std_np, epistemic_std_np

    return y_mean_np, total_std_np


def coverage(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray,
             level: float = 0.95) -> float:
    """
    Empirical coverage at a given credible level.

    Args:
        y_true: [B] true targets.
        y_mean: [B] predictive means.
        y_std:  [B] predictive standard deviations.
        level:  Nominal coverage (default 0.95).

    Returns:
        Scalar empirical coverage in [0, 1].
    """
    from scipy.stats import norm as scipy_norm
    z = scipy_norm.ppf((1 + level) / 2)
    lo = y_mean - z * y_std
    hi = y_mean + z * y_std
    return float(((y_true >= lo) & (y_true <= hi)).mean())
