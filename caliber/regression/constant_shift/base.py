from typing import Callable

import numpy as np

from caliber.regression.base import AbstractRegressionModel


class ConstantQuantileShiftRegressionModel(AbstractRegressionModel):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        n_bins: int = 100,
        step_size: float = 1.0,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.n_bins = n_bins
        self.step_size = step_size

    def fit(self, quantiles: np.ndarray, targets: np.ndarray) -> None:
        bins = self._get_bin_edges(quantiles, targets)
        self._params = bins[
            np.argmin(
                [
                    self.loss_fn(targets, self._predict(params, quantiles))
                    for params in bins
                ]
            )
        ]

    def predict(self, quantiles: np.ndarray) -> np.ndarray:
        return self._predict(self._params, quantiles)

    def _get_bin_edges(self, quantiles: np.ndarray, targets: np.ndarray) -> np.ndarray:
        diff = targets - quantiles
        return np.linspace(np.min(diff), np.max(diff), self.n_bins + 1)

    def _predict(self, params: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        return quantiles + self.step_size * params
