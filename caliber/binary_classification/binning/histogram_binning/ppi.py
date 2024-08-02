import numpy as np

from caliber.binary_classification.binning.histogram_binning.base import (
    HistogramBinningBinaryClassificationModel,
)


class PPIHistogramBinningBinaryClassificationModel(
    HistogramBinningBinaryClassificationModel
):
    def fit(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        pseudo_targets: np.ndarray,
        labeled_indices: np.ndarray,
    ):
        self._bin_edges = self._get_bin_edges()
        bin_indices = np.digitize(probs, self._bin_edges)
        self._params = []

        is_labeled = np.zeros(len(targets), dtype=bool)
        is_labeled[labeled_indices] = True

        var_pseudo_targets = np.var(pseudo_targets)
        pseudo_targets_size = len(pseudo_targets)
        targets_size = len(labeled_indices)
        if var_pseudo_targets > 0:
            cov_true_pseudo_targets = np.cov(targets[is_labeled], pseudo_targets[is_labeled])[0, 1]
            lam = cov_true_pseudo_targets / (
                    (1 + targets_size / (pseudo_targets_size - targets_size)) * var_pseudo_targets
            )
        else:
            lam = 1

        for i in range(1, self.n_bins + 2):
            mask = bin_indices == i
            self._fit_bin(i, mask, probs, targets, pseudo_targets, is_labeled, lam)

    def _fit_bin(
        self,
        i: int,
        mask: np.ndarray,
        probs: np.ndarray,
        targets: np.ndarray,
        pseudo_targets: np.ndarray,
        is_labeled: np.ndarray,
        lam: float
    ):
        prob_bin = np.mean(mask)
        masked_targets = targets[mask]
        masked_probs = probs[mask]
        masked_pseudo_targets = lam * pseudo_targets[mask]
        masked_is_labeled = is_labeled[mask]

        self._params.append(
            (
                np.mean(masked_pseudo_targets[~masked_is_labeled])
                + np.mean(
                    masked_targets[masked_is_labeled]
                    - masked_pseudo_targets[masked_is_labeled]
                )
                - np.mean(masked_probs)
                if len(masked_pseudo_targets[~masked_is_labeled]) > 0
                else np.mean(masked_targets[masked_is_labeled]) - np.mean(masked_probs)
            )
            if prob_bin >= self.min_prob_bin
            else np.nan
        )
