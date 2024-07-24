import numpy as np

from caliber.binary_classification.binning.histogram_binning.base import HistogramBinningBinaryClassificationModel


class PPIHistogramBinningBinaryClassificationModel(HistogramBinningBinaryClassificationModel):

    def fit(self, probs: np.ndarray, targets: np.ndarray, pseudo_indices: np.ndarray):
        self._bin_edges = self._get_bin_edges()
        bin_indices = np.digitize(probs, self._bin_edges)
        self._params = []

        is_pseudo = np.zeros(len(targets), dtype=bool)
        is_pseudo[pseudo_indices] = True

        for i in range(1, self.n_bins + 2):
            mask = bin_indices == i
            self._fit_bin(i, mask, probs, targets, is_pseudo)

    def _fit_bin(
        self, i: int, mask: np.ndarray, probs: np.ndarray, targets: np.ndarray, is_pseudo: np.ndarray
    ):
        prob_bin = np.mean(mask)
        masked_targets = targets[mask]
        masked_is_pseudo = is_pseudo[mask]

        self._params.append(
            np.mean(masked_targets[masked_is_pseudo])\
            + np.mean(masked_targets[~masked_is_pseudo] - masked_targets[~masked_is_pseudo])
            if prob_bin >= self.min_prob_bin else np.nan
        )
