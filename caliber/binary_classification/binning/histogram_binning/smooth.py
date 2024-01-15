from typing import Optional

import numpy as np
from scipy.stats import norm


class SmoothHistogramBinningBinaryClassificationModel:
    def __init__(
        self,
        n_bins: int = 10,
        split: float = 0.8,
        seed: int = 0,
        smoothness: float = 0.1,
    ):
        self._n_bins = n_bins
        self._rng = np.random.default_rng(seed)
        self.split = split
        self._params = None
        self._bin_edges = None
        self.smoothness = smoothness

    def fit(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ):
        if groups is None:
            groups = self._initialize_groups(len(probs))
        self._bin_edges = self._get_bin_edges()
        kernels = self._get_kernels(probs, self.smoothness)
        A = np.mean(
            kernels[:, :, None, None, None]
            * kernels[:, None, None, :, None]
            * groups[:, None, :, None, None]
            * groups[:, None, None, None],
            0,
        ).reshape((self._n_bins + 1) * groups.shape[1], -1)
        b = np.mean(
            kernels[:, :, None] * groups[:, None] * (targets - probs)[:, None, None], 0
        ).flatten()
        self._params = np.linalg.solve(A, b).reshape(self._n_bins + 1, groups.shape[1])

    def predict_proba(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        probs = np.copy(probs)
        if groups is None:
            groups = self._initialize_groups(len(probs))
        kernels = self._get_kernels(probs, self.smoothness)
        probs += np.sum(self._params * kernels[:, :, None] * groups[:, None], (1, 2))
        return np.clip(probs, 0, 1)

    def predict(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return (self.predict_proba(probs, groups) >= 0.5).astype(int)

    def _get_bin_edges(self):
        return np.linspace(0, 1, self._n_bins + 1)

    def _get_kernels(self, probs: np.ndarray, smoothness) -> np.ndarray:
        return np.stack([norm.pdf(probs, i, smoothness) for i in self._bin_edges]).T

    @staticmethod
    def _initialize_groups(size: int):
        return np.ones((size, 1)).astype(bool)
