import torch
import pandas as pd
import numpy as np
import itertools


class SubsetMaker:
    """
    Builds a powerset lookup table from a training DataFrame.

    For each non-empty subset of `subset_features`, groups training examples by
    their unique value combination and stores the empirical mean and variance of
    the target within each group.

    Args:
        df (pd.DataFrame): Training data. All `subset_features` columns must be
            integer-coded categorical variables.
        subset_features (list[str]): Names of the categorical feature columns.
        target (list[str]): Single-element list with the target column name.
    """

    def __init__(self, df: pd.DataFrame, subset_features: list, target: list):
        self.df = df
        self.subset_features = subset_features
        self.target = target
        self.lookup = self._build_lookup()
        valid_target = self.df[self.target[0]].dropna()
        self.fallback_mean = float(valid_target.mean())
        self.fallback_var = float(valid_target.var())

    def _get_powerset(self) -> list:
        powerset = []
        for r in range(1, len(self.subset_features) + 1):
            powerset += list(itertools.combinations(self.subset_features, r))
        return [list(s) for s in powerset]

    def _build_lookup(self, drop_missing_rows: bool = True) -> dict:
        lookup = {}
        for subset in self._get_powerset():
            grouped = (
                self.df[subset + self.target]
                .groupby(subset)
                .agg({self.target[0]: ["mean", "var"]})
                .reset_index()
            )
            grouped.columns = [
                "_".join(c).strip("_") if isinstance(c, tuple) else c
                for c in grouped.columns
            ]
            if drop_missing_rows:
                grouped = grouped.dropna()
                grouped = grouped[grouped[f"{self.target[0]}_var"] > 1e-6]

            mean_col = f"{self.target[0]}_mean"
            var_col = f"{self.target[0]}_var"
            group_dict = {
                tuple(row[subset]): (row[mean_col], row[var_col])
                for _, row in grouped.iterrows()
            }
            lookup[tuple(subset)] = (subset, grouped, group_dict)
        return lookup

    def batch_lookup(self, batch_tensor: torch.Tensor):
        """
        Look up empirical means/variances for a batch of integer-coded examples.

        Args:
            batch_tensor: [B, D] integer tensor matching the order of subset_features.

        Returns:
            means:     [B, |S|] float tensor
            variances: [B, |S|] float tensor
            mask:      [B, |S|] bool tensor (True where the cell exists in training data)
        """
        batch_df = pd.DataFrame(
            batch_tensor.numpy(), columns=self.subset_features
        ).astype(int)

        all_means, all_vars, all_masks = [], [], []

        for _, (subset_cols, _, group_dict) in self.lookup.items():
            subset_vals = batch_df[subset_cols].apply(tuple, axis=1)
            means, vars_, mask = [], [], []
            for key in subset_vals:
                if key in group_dict:
                    m, v = group_dict[key]
                    means.append(m)
                    vars_.append(v)
                    mask.append(True)
                else:
                    means.append(self.fallback_mean)
                    vars_.append(self.fallback_var)
                    mask.append(False)
            all_means.append(means)
            all_vars.append(vars_)
            all_masks.append(mask)

        # Transpose from [|S|, B] → [B, |S|]
        return (
            torch.tensor(all_means, dtype=torch.float32).T,
            torch.tensor(all_vars, dtype=torch.float32).T,
            torch.tensor(all_masks, dtype=torch.bool).T,
        )
