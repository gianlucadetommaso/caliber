import abc

import numpy as np


class BinningBinaryClassificationModel:
    def __init__(self, n_bins: int = 10):
        self._n_bins = n_bins
        self._params = None
        self._bin_edges = None

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        self._bin_edges = self._get_bin_edges()
        bin_indices = np.digitize(probs, self._bin_edges)
        self._params = []

        for i in range(1, self._n_bins + 2):
            mask = bin_indices == i
            self._fit_bin(i, mask, probs, targets)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if self._bin_edges is None:
            raise ValueError("Run `fit` first.")
        bin_indices = np.digitize(probs, self._bin_edges)
        probs = np.copy(probs)

        for i in range(1, self._n_bins + 2):
            mask = bin_indices == i
            probs[mask] = self._predict_bin(i, mask, probs)
        return probs

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs) >= 0.5).astype(int)

    def _get_bin_edges(self):
        return np.linspace(0, 1, self._n_bins + 1)

    @abc.abstractmethod
    def _fit_bin(
        self, i: int, mask: np.ndarray, probs: np.ndarray, targets: np.ndarray
    ):
        pass

    @abc.abstractmethod
    def _predict_bin(self, i: int, mask: np.ndarray, probs: np.ndarray):
        pass
