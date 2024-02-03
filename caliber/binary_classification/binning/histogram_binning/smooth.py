import logging
from typing import Optional

import numpy as np
from scipy.stats import norm

from caliber.binary_classification.metrics.asce import (
    average_smooth_squared_calibration_error,
)


class IterativeSmoothHistogramBinningBinaryClassificationModel:
    def __init__(
        self,
        n_bins: int = 10,
        split: float = 0.8,
        seed: int = 0,
        smoothness: float = 0.1,
        max_rounds: int = 1000,
    ):
        self.n_bins = n_bins
        self._rng = np.random.default_rng(seed)
        self.split = split
        self._params = None
        self._bin_edges = None
        self.smoothness = smoothness
        self.max_rounds = max_rounds

    def fit(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ):
        if groups is None:
            groups = self._initialize_groups(len(probs))
        self._bin_edges = self._get_bin_edges()
        self._params = []

        n_data = len(probs)
        perm = self._rng.choice(n_data, n_data, replace=False)
        calib_size = int(np.ceil(n_data * self.split))
        calib_probs, val_probs = probs[perm[:calib_size]], probs[perm[calib_size:]]
        calib_targets, val_targets = (
            targets[perm[:calib_size]],
            targets[perm[calib_size:]],
        )
        calib_groups, val_groups = (
            groups[perm[:calib_size]],
            groups[perm[calib_size:]],
        )

        val_assces = [
            average_smooth_squared_calibration_error(
                val_targets, val_probs, smoothness=self.smoothness
            )
        ]

        for t in range(self.max_rounds):
            params = self._get_params(calib_probs, calib_targets, calib_groups)

            val_probs = self._update_proba(params, val_probs, val_groups)
            val_assces.append(
                average_smooth_squared_calibration_error(
                    val_targets, val_probs, smoothness=self.smoothness
                )
            )

            if val_assces[-1] >= val_assces[-2]:
                logging.info(
                    f"Early stopping triggered after {t} rounds. The ASSCE started increasing on the validation data."
                )
                break

            self._params.append(params)
            calib_probs = self._update_proba(params, calib_probs, calib_groups)

        return dict(n_iter=len(val_assces) - 1, val_losses=val_assces)

    def predict_proba(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        probs = np.copy(probs)
        if groups is None:
            groups = self._initialize_groups(len(probs))
        for params in self._params:
            probs = self._update_proba(params, probs, groups)
        return probs

    def predict(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return (self.predict_proba(probs, groups) >= 0.5).astype(int)

    def _update_proba(
        self, params: np.ndarray, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        probs = np.copy(probs)
        if groups is None:
            groups = self._initialize_groups(len(probs))
        kernels = self._get_kernels(probs, self.smoothness)
        probs += np.sum(params * kernels[:, :, None] * groups[:, None], (1, 2))
        return np.clip(probs, 0, 1)

    def _get_bin_edges(self):
        return np.linspace(0, 1, self.n_bins + 1)

    def _get_kernels(self, probs: np.ndarray, smoothness) -> np.ndarray:
        return np.stack([norm.pdf(probs, i, smoothness) for i in self._bin_edges]).T

    @staticmethod
    def _initialize_groups(size: int):
        return np.ones((size, 1)).astype(bool)

    def _get_params(
        self, probs: np.ndarray, targets: np.ndarray, groups: np.ndarray
    ) -> np.ndarray:
        kernels = self._get_kernels(probs, self.smoothness)
        A = np.mean(
            kernels[:, :, None, None, None]
            * kernels[:, None, None, :, None]
            * groups[:, None, :, None, None]
            * groups[:, None, None, None],
            0,
        ).reshape((self.n_bins + 1) * groups.shape[1], -1)
        b = np.mean(
            kernels[:, :, None] * groups[:, None] * (targets - probs)[:, None, None], 0
        ).flatten()
        return np.linalg.lstsq(A, b, rcond=None)[0].reshape(
            self.n_bins + 1, groups.shape[1]
        )
