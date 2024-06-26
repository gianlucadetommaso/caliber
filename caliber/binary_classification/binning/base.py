import abc

import numpy as np

from caliber.binary_classification.base import AbstractBinaryClassificationModel
from caliber.binary_classification.pred_from_probs_mixin import (
    PredFromProbsBinaryClassificationMixin,
)


class BinningBinaryClassificationModel(
    PredFromProbsBinaryClassificationMixin, AbstractBinaryClassificationModel
):
    def __init__(self, n_bins: int = 10, min_prob_bin: float = 0.0):
        super().__init__(threshold=0.5)
        self.min_prob_bin = min_prob_bin
        self.n_bins = n_bins
        self._bin_edges = None

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        self._bin_edges = self._get_bin_edges()
        bin_indices = np.digitize(probs, self._bin_edges)
        self._params = []

        for i in range(1, self.n_bins + 2):
            mask = bin_indices == i
            self._fit_bin(i, mask, probs, targets)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if self._bin_edges is None:
            raise ValueError("Run `fit` first.")
        bin_indices = np.digitize(probs, self._bin_edges)
        probs = np.copy(probs)

        for i in range(1, self.n_bins + 2):
            mask = bin_indices == i
            probs[mask] = self._predict_bin(i, mask, probs)
        return probs

    def _get_bin_edges(self):
        return np.linspace(0, 1, self.n_bins + 1)

    @abc.abstractmethod
    def _fit_bin(
        self, i: int, mask: np.ndarray, probs: np.ndarray, targets: np.ndarray
    ):
        pass

    @abc.abstractmethod
    def _predict_bin(self, i: int, mask: np.ndarray, probs: np.ndarray):
        pass
