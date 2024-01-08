import numpy as np
from caliber.binary_classification.metrics.asce import average_squared_calibration_error
from scipy.optimize import brute


class IterativeHistogramBinningBinaryClassificationModel:
    def __init__(self, n_bins: int = 100, max_rounds: int = 100, min_prob_bin: float = 0.):
        self._n_bins = n_bins
        self._max_rounds = max_rounds
        self._params = None
        self._bin_edges = None
        self._min_prob_bin = 0

    @staticmethod
    def _loss_fn(i: int, bin_indices: np.ndarray, targets: np.ndarray, probs: np.ndarray) -> float:
        mask = bin_indices == i
        return average_squared_calibration_error(targets[mask], probs[mask]) if np.sum(mask) else 0.

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        self._bin_edges = self._get_bin_edges()
        bin_indices = np.digitize(probs, self._bin_edges)
        self._params = []
        probs = np.copy(probs)

        for t in range(self._max_rounds):
            i = np.argmin([self._loss_fn(i, bin_indices, targets, probs) for i in range(1, self._n_bins + 1)])
            mask = bin_indices == i
            prob_bin = np.mean(mask)
            if prob_bin <= self._min_prob_bin:
                break
            patch = np.mean(targets[mask])
            self._params.append((i, patch)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if self._bin_edges is None:
            raise ValueError("Run `fit` first.")
        bin_indices = np.digitize(probs, self._bin_edges)
        probs = np.copy(probs)

        for i, _params in enumerate(self._params):
            mask = bin_indices == i + 1
            if not np.isnan(_params):
                probs[mask] = _params
        return probs

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs) >= 0.5).astype(int)

    def _get_bin_edges(self):
        return np.linspace(0, 1, self._n_bins + 1)
