from typing import Optional

import numpy as np

from caliber.binary_classification.minimizing.linear_scaling.performance.base import (
    PerformanceLinearScalingBinaryClassificationModel,
)


class PredictiveValuesLinearScalingBinaryClassificationModel(
    PerformanceLinearScalingBinaryClassificationModel
):
    def __init__(
        self,
        threshold: float,
        lam: float = 0.01,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        num_features: int = 0,
    ):
        super().__init__(
            loss_fn=_predictive_values_loss_fn,
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            num_features=num_features,
        )
        self._lam = lam


def _predictive_values_loss_fn(targets: np.ndarray, preds: np.ndarray) -> float:
    n_pos_preds = np.sum(preds)
    n_neg_preds = len(preds) - n_pos_preds
    ppv = np.sum(targets * preds) / n_pos_preds if n_pos_preds > 0 else 0.0
    npv = np.sum((1 - targets) * (1 - preds)) / n_neg_preds if n_neg_preds > 0 else 0.0
    return -ppv * npv / (ppv + npv)
