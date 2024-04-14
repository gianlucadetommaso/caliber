from typing import Literal

import numpy as np

from caliber.regression.conformal_regression.base import ConformalRegressionModel


class ConformalizedQuantileRegressionModel(ConformalRegressionModel):
    def __init__(
        self,
        confidence: float,
        interval_type: Literal[
            "two-tailed", "left-tailed", "right-tailed"
        ] = "two-tailed",
    ):
        self.confidence = confidence
        self.interval_type = interval_type
        self._params = None

    def fit(self, quantiles: np.ndarray, targets: np.ndarray) -> None:
        size = len(quantiles)
        adjusted_confidence = np.ceil((size + 1) * self.confidence) / size
        if self.interval_type == "two-tailed":
            self._get_two_tailed_error(quantiles)
            scores = np.maximum(quantiles[:, 0] - targets, targets - quantiles[:, 1])
        elif self.interval_type == "left-tailed":
            self._get_left_tailed_error(quantiles)
            scores = quantiles - targets
        elif self.interval_type == "right-tailed":
            self._get_right_tailed_error(quantiles)
            scores = targets - quantiles
        else:
            self._get_interval_type_error()
        self._params = np.quantile(scores, adjusted_confidence)

    def predict_interval(self, quantiles: np.ndarray) -> np.ndarray:
        if self.interval_type == "two-tailed":
            self._get_two_tailed_error(quantiles)
            lowers = quantiles[:, 0] - self._params
            uppers = quantiles[:, 1] + self._params
            bounds = np.stack((lowers, uppers), axis=1)
        elif self.interval_type == "left-tailed":
            self._get_left_tailed_error(quantiles)
            bounds = quantiles - self._params
        elif self.interval_type == "right-tailed":
            self._get_right_tailed_error(quantiles)
            bounds = quantiles + self._params
        else:
            self._get_interval_type_error()
        return bounds

    @staticmethod
    def _get_two_tailed_error(quantiles: np.ndarray) -> None:
        if quantiles.ndim != 2 or quantiles.shape[1] != 2:
            raise ValueError(
                "`quantiles` must be a two-dimensional array of quantiles with `quantiles.shape[1]` equal 2."
            )

    @staticmethod
    def _get_left_tailed_error(quantiles: np.ndarray) -> None:
        if quantiles.ndim != 1:
            raise ValueError(
                "`quantiles` must be a one-dimensional array of right quantiles."
            )

    @staticmethod
    def _get_right_tailed_error(quantiles: np.ndarray) -> None:
        if quantiles.ndim != 1:
            raise ValueError(
                "`quantiles` must be a one-dimensional array of left quantiles."
            )

    def _get_interval_type_error(self):
        raise ValueError(f"`interval_type={self.interval_type}` not recognized.")
