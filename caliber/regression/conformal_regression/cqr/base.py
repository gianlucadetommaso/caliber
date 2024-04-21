from typing import Literal

import numpy as np

from caliber.regression.base import AbstractRegressionModel
from caliber.utils.interval_type_error import interval_type_error
from caliber.utils.quantile_checks import (
    left_tailed_quantile_check,
    right_tailed_quantile_check,
    two_tailed_quantile_check,
)


class ConformalizedQuantileRegressionModel(AbstractRegressionModel):
    def __init__(
        self,
        confidence: float,
        interval_type: Literal[
            "two-tailed", "left-tailed", "right-tailed"
        ] = "two-tailed",
    ):
        super().__init__()
        self.confidence = confidence
        self.interval_type = interval_type
        self._params = None

    def fit(self, quantiles: np.ndarray, targets: np.ndarray) -> None:
        size = len(quantiles)
        adjusted_confidence = np.ceil((size + 1) * self.confidence) / size
        if self.interval_type == "two-tailed":
            two_tailed_quantile_check(quantiles)
            scores = np.maximum(quantiles[:, 0] - targets, targets - quantiles[:, 1])
        elif self.interval_type == "left-tailed":
            left_tailed_quantile_check(quantiles)
            scores = quantiles - targets
        elif self.interval_type == "right-tailed":
            right_tailed_quantile_check(quantiles)
            scores = targets - quantiles
        else:
            interval_type_error(self.interval_type)
        self._params = np.quantile(scores, adjusted_confidence)

    def predict(self, quantiles: np.ndarray) -> np.ndarray:
        if self.interval_type == "two-tailed":
            two_tailed_quantile_check(quantiles)
            lowers = quantiles[:, 0] - self._params
            uppers = quantiles[:, 1] + self._params
            bounds = np.stack((lowers, uppers), axis=1)
        elif self.interval_type == "left-tailed":
            left_tailed_quantile_check(quantiles)
            bounds = quantiles - self._params
        elif self.interval_type == "right-tailed":
            right_tailed_quantile_check(quantiles)
            bounds = quantiles + self._params
        else:
            interval_type_error(self.interval_type)
        return bounds
