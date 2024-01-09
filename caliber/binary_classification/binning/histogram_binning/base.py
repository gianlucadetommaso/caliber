import numpy as np

from caliber.binary_classification.binning.base import BinningBinaryClassificationModel


class HistogramBinningBinaryClassificationModel(BinningBinaryClassificationModel):
    def _fit_bin(
        self, i: int, mask: np.ndarray, probs: np.ndarray, targets: np.ndarray
    ):
        prob_bin = np.mean(mask)
        self._params.append(np.mean(targets[mask]) if prob_bin > 0 else np.nan)

    def _predict_bin(self, i: int, mask: np.ndarray, probs: np.ndarray):
        if not np.isnan(self._params[i - 1]):
            return self._params[i - 1]
        return probs[mask]
