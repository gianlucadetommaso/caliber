import numpy as np

from caliber.binary_classification.binning.base import BinningBinaryClassificationModel


class IsotonicRegressionBinaryClassificationModel(BinningBinaryClassificationModel):
    def _fit_bin(
        self, i: int, mask: np.ndarray, probs: np.ndarray, targets: np.ndarray
    ):
        prob_bin = np.mean(mask)
        if prob_bin > 0:
            if i == 1:
                self._params.append(np.mean(targets[mask]))
            else:
                self._params.append(max(self._params[-1], np.mean(targets[mask])))
        else:
            if i == 1:
                self._params.append(self._bin_edges[1] - self._bin_edges[0])
            else:
                self._params.append(self._params[-1])

    def _predict_bin(self, i: int, mask: np.ndarray, probs: np.ndarray):
        return self._params[i - 1]
