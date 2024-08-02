from typing import Literal

import numpy as np

from caliber.regression.conformal_regression.base import (
    ConformalizedScoreRegressionModel,
)
from caliber.utils.quantile_checks import (
    both_quantile_check,
    lower_quantile_check,
    upper_quantile_check,
)
from caliber.utils.quantile_error import which_quantile_error


class ConformalizedQuantileRegressionModel(ConformalizedScoreRegressionModel):
    def __init__(
        self,
        confidence: float,
        which_quantile: Literal["both", "lower", "upper"] = "both",
    ):
        super().__init__(confidence=confidence)
        self.which_quantile = which_quantile

    def fit(self, quantiles: np.ndarray, targets: np.ndarray) -> None:
        if self.which_quantile == "both":
            both_quantile_check(quantiles)
            scores = np.maximum(quantiles[:, 0] - targets, targets - quantiles[:, 1])
        elif self.which_quantile == "lower":
            lower_quantile_check(quantiles)
            scores = quantiles - targets
        elif self.which_quantile == "upper":
            upper_quantile_check(quantiles)
            scores = targets - quantiles
        else:
            which_quantile_error(self.which_quantile)
        super().fit(scores, targets)

    def predict(self, quantiles: np.ndarray) -> np.ndarray:
        if self.which_quantile == "both":
            both_quantile_check(quantiles)
            lowers = quantiles[:, 0] - self._params
            uppers = quantiles[:, 1] + self._params
            bounds = np.stack((lowers, uppers), axis=1)
        elif self.which_quantile == "lower":
            lower_quantile_check(quantiles)
            bounds = quantiles - self._params
        elif self.which_quantile == "upper":
            upper_quantile_check(quantiles)
            bounds = quantiles + self._params
        else:
            which_quantile_error(self.which_quantile)
        return bounds
