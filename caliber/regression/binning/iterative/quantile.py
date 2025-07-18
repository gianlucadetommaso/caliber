from typing import Optional

import numpy as np

from caliber.regression.binning.iterative.base import IterativeBinningRegressionModel
from caliber.utils.quantile_checks import both_quantile_check, single_quantile_check
from caliber.utils.quantile_error import which_quantile_error


class IterativeBinningQuantileRegressionModel(IterativeBinningRegressionModel):
    def fit(
        self,
        quantiles: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> dict:
        self._check_quantiles(quantiles)
        return super().fit(values=quantiles, targets=targets, groups=groups)

    def predict(
        self, quantiles: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self._check_quantiles(quantiles)
        return super().predict(values=quantiles, groups=groups)

    def _get_scores(self, quantiles: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if self.which_quantile == "both":
            scores = np.maximum(quantiles[:, 0] - targets, targets - quantiles[:, 1])
        elif self.which_quantile == "lower":
            scores = quantiles - targets
        elif self.which_quantile == "upper":
            scores = targets - quantiles
        return scores

    def _get_inverse_scores(
        self, quantiles: np.ndarray, score_quantiles: np.ndarray
    ) -> np.ndarray:
        if self.which_quantile == "both":
            quantiles += np.array([-1, 1]) * score_quantiles[:, None]
        elif self.which_quantile == "upper":
            quantiles += score_quantiles
        else:
            quantiles -= score_quantiles
        return quantiles

    def _check_quantiles(self, quantiles: np.ndarray) -> None:
        if self.which_quantile == "both":
            both_quantile_check(quantiles, y_dim=1)
        elif self.which_quantile == "lower":
            single_quantile_check(quantiles, y_dim=1)
        elif self.which_quantile == "upper":
            single_quantile_check(quantiles, y_dim=1)
        else:
            which_quantile_error(quantiles)
