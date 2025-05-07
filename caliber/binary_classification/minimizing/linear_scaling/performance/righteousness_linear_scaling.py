from typing import Optional

import numpy as np

from caliber.binary_classification.minimizing.linear_scaling.performance.base import (
    PerformanceLinearScalingBinaryClassificationModel,
)


class RighteousnessLinearScalingBinaryClassificationModel(
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
            loss_fn=_righteousness_loss_fn,
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            num_features=num_features,
        )
        self._lam = lam


def _righteousness_loss_fn(targets: np.ndarray, preds: np.ndarray) -> float:
    p11 = np.sum(targets * preds)
    p00 = np.sum((1 - targets) * (1 - preds))
    p01 = np.sum((1 - targets) * preds)
    p10 = np.sum(targets * (1 - preds))
    return -(1 - p01 * p10) * p00 * p11
