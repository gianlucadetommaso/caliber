from typing import Optional

import numpy as np

from caliber.regression.binning.iterative.base import IterativeBinningRegressionModel


class IterativeBinningMeanRegressionModel(IterativeBinningRegressionModel):
    def fit(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> dict:
        return super().fit(values=preds, targets=targets, groups=groups)

    def predict(
        self, preds: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return super().predict(values=preds, groups=groups)

    def _get_scores(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.abs(targets - preds)

    def _get_inverse_scores(
        self, preds: np.ndarray, score_quantiles: np.ndarray
    ) -> np.ndarray:
        if self.which_quantile == "both":
            quantiles = np.stack(
                (preds - score_quantiles, preds + score_quantiles), axis=1
            )
        elif self.which_quantile == "upper":
            quantiles = preds + score_quantiles
        else:
            quantiles = preds - score_quantiles
        return quantiles
