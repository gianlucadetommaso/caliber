from typing import Literal

import numpy as np
from numpy.typing import NDArray

from caliber.regression.conformal_regression.base import (
    ConformalizedScoreRegressionModel,
)
from caliber.utils.functional import maybe_squeeze
from caliber.utils.quantile_checks import both_quantile_check, single_quantile_check
from caliber.utils.quantile_error import which_quantile_error


class ConformalizedQuantileRegressionModel(ConformalizedScoreRegressionModel):
    def __init__(
        self,
        confidence: float,
        which_quantile: Literal["both", "lower", "upper"] = "both",
    ):
        super().__init__(confidence=confidence)
        self.which_quantile = which_quantile

    def fit(self, quantiles: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        if targets.ndim == 1:
            targets = targets[:, None]
        if quantiles.ndim == 1:
            quantiles = quantiles[:, None]
        self._y_dim = targets.shape[1]

        if self.which_quantile == "both":
            both_quantile_check(quantiles, self._y_dim)
            lowers, uppers = quantiles[:, : self._y_dim], quantiles[:, self._y_dim :]
            scores = np.maximum(lowers - targets, targets - uppers)
        elif self.which_quantile == "lower":
            single_quantile_check(quantiles, self._y_dim)
            scores = quantiles - targets
        elif self.which_quantile == "upper":
            single_quantile_check(quantiles, self._y_dim)
            scores = targets - quantiles
        else:
            which_quantile_error(self.which_quantile)
        super().fit(scores, targets)

    def predict(self, quantiles: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.which_quantile == "both":
            both_quantile_check(quantiles, self._y_dim)
            lowers, uppers = quantiles[:, : self._y_dim], quantiles[:, self._y_dim :]
            return np.concatenate(
                (lowers - self._params, uppers + self._params), axis=1
            )

        elif self.which_quantile == "lower":
            single_quantile_check(quantiles, self._ydim)
            return maybe_squeeze(quantiles - self._params, 1)

        elif self.which_quantile == "upper":
            single_quantile_check(quantiles, self._ydim)
            return maybe_squeeze(quantiles + self._params, 1)

        which_quantile_error(self.which_quantile)
