import torch
import pandas as pd
import numpy as np
from .subset_maker import SubsetMaker


class SubsetMixturePredictor:
    """
    Inference wrapper for a trained Subset Mixture Model.

    Args:
        subset_maker (SubsetMaker): Trained SubsetMaker with lookup table.
        weight_vector (torch.Tensor): Softmaxed mixture weights [|S|].
    """

    def __init__(self, subset_maker: SubsetMaker, weight_vector: torch.Tensor):
        self.subset_maker = subset_maker
        self.weight_vector = weight_vector

    def predict(self, df: pd.DataFrame, return_debug: bool = False):
        """
        Predict target values for a new DataFrame.

        For each example, weights are masked to valid subsets (those seen during
        training) and re-normalized. Examples with no valid subsets fall back to
        the global training mean.

        Args:
            df: Input DataFrame containing all subset_features columns.
            return_debug: If True, also return per-example normalized weights
                and a fallback mask.

        Returns:
            preds (np.ndarray): Predicted values [B].
            (optional) norm_weights (np.ndarray): [B, |S|]
            (optional) fallback_mask (np.ndarray): [B] bool
        """
        batch_tensor = torch.tensor(
            df[self.subset_maker.subset_features].astype(np.float32).values
        )
        mus, _, mask = self.subset_maker.batch_lookup(batch_tensor)  # [B, S]

        B, S = mus.shape
        weights = self.weight_vector.unsqueeze(0).expand(B, S)       # [B, S]
        weights = weights.masked_fill(~mask, 0.0)
        mus = mus.masked_fill(~mask, 0.0)

        weight_sums = weights.sum(dim=1, keepdim=True)               # [B, 1]
        norm_weights = weights / (weight_sums + 1e-9)

        fallback_mask = weight_sums.squeeze() < 1e-6                 # [B]
        pred_weighted = (mus * norm_weights).sum(dim=1)
        pred_fallback = torch.full_like(
            pred_weighted, self.subset_maker.fallback_mean
        )
        preds = torch.where(fallback_mask, pred_fallback, pred_weighted)

        if return_debug:
            return preds.numpy(), norm_weights.numpy(), fallback_mask.numpy()
        return preds.numpy()
