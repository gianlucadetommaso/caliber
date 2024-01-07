from typing import Optional

import numpy as np

from caliber.binary_classification.linear_scaling.performance.base import (
    CustomPerformanceBinaryClassificationLinearScaling,
)


class PredictiveValuesBinaryClassificationLinearScaling(
    CustomPerformanceBinaryClassificationLinearScaling
):
    def __init__(
        self,
        threshold: float,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
    ):
        super().__init__(
            loss_fn=predictive_values_loss_fn,
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )


def predictive_values_loss_fn(preds: np.ndarray, targets: np.ndarray) -> float:
    n_pos_preds = np.sum(preds)
    n_neg_preds = len(preds) - n_pos_preds
    ppv = np.sum(targets * preds) / n_pos_preds if n_pos_preds > 0 else 0.0
    npv = np.sum((1 - targets) * (1 - preds)) / n_neg_preds if n_neg_preds > 0 else 0.0
    return -ppv * npv / (ppv + npv)
