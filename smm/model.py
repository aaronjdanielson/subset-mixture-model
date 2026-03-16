import numpy as np
import torch
import torch.nn.functional as F


class SubsetWeightsModel(torch.nn.Module):
    """
    Learns a single global weight vector over all feature subsets.

    A real-valued parameter vector eta of length |S| is initialized to zero and
    passed through softmax to produce the mixture weights pi.

    Args:
        num_subsets (int): Number of subsets in the powerset (|S|).
    """

    def __init__(self, num_subsets: int):
        super().__init__()
        self.eta = torch.nn.Parameter(torch.zeros(num_subsets))

    def forward(self, x=None):
        # Return raw logits; softmax is applied inside the loss function.
        return self.eta


class SubsetDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping a DataFrame of integer-coded categorical features.

    Args:
        df (pd.DataFrame): Data split (train / val / test).
        subset_features (list[str]): Categorical feature column names.
        target (list[str]): Single-element list with the target column name.
    """

    def __init__(self, df, subset_features: list, target: list):
        self.df = df
        self.subset_features = subset_features
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.subset_features].astype(np.float32).values)
        y = torch.tensor(np.float32(row[self.target[0]]))
        return x, y


def subset_mixture_neg_log_posterior(
    pi_logits: torch.Tensor,
    y: torch.Tensor,
    mus: torch.Tensor,
    variances: torch.Tensor,
    mask: torch.Tensor = None,
    alpha: float = 1.1,
) -> torch.Tensor:
    """
    Negative log-posterior of the subset mixture model.

    Loss = -sum_i log( sum_s pi_s * N(y_i | mu_s(x_i), sigma^2_s(x_i)) )
           - (alpha - 1) * sum_s log(pi_s)

    Args:
        pi_logits: [|S|] unnormalized logits (global parameter).
        y:         [B] target values.
        mus:       [B, |S|] empirical conditional means.
        variances: [B, |S|] empirical conditional variances.
        mask:      [B, |S|] bool tensor; True where the cell exists in training data.
        alpha:     Dirichlet concentration parameter (>1 encourages non-degenerate weights).

    Returns:
        Scalar loss tensor.
    """
    pi = F.softmax(pi_logits, dim=0)          # [|S|]
    log_pi = torch.log(pi + 1e-9)
    log_pi_b = log_pi.unsqueeze(0)            # [1, |S|]

    log_probs = (
        -0.5 * torch.log(2 * torch.pi * variances)
        - 0.5 * (y.unsqueeze(1) - mus) ** 2 / variances
    )                                          # [B, |S|]

    log_weighted = log_probs + log_pi_b        # [B, |S|]

    if mask is not None:
        log_weighted = log_weighted.masked_fill(~mask, float("-inf"))

    log_likelihoods = torch.logsumexp(log_weighted, dim=1)  # [B]
    nll = -log_likelihoods.sum()

    batch_size = y.size(0)
    log_prior = (alpha - 1.0) * log_pi.sum()

    return nll - log_prior / batch_size


def subset_mixture_mse(
    pi_logits: torch.Tensor,
    y: torch.Tensor,
    mus: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    MSE loss for the subset mixture model (useful for warmup / debugging).

    Args:
        pi_logits: [|S|] unnormalized logits.
        y:         [B] target values.
        mus:       [B, |S|] empirical conditional means.
        mask:      [B, |S|] bool tensor.

    Returns:
        Scalar MSE tensor.
    """
    pi = F.softmax(pi_logits, dim=0)           # [|S|]
    weights = pi.unsqueeze(0).expand_as(mus)   # [B, |S|]

    if mask is not None:
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

    preds = (mus * weights).sum(dim=1)
    return F.mse_loss(preds, y, reduction="sum")
