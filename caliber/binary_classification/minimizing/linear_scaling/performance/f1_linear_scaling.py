from typing import Optional

import numpy as np

from caliber.binary_classification.minimizing.linear_scaling.performance.base import (
    PerformanceLinearScalingBinaryClassificationModel,
)


class PositiveF1LinearScalingBinaryClassificationModel(
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
            loss_fn=_pos_f1_loss_fn,
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            num_features=num_features,
        )
        self._lam = lam


class NegativeF1LinearScalingBinaryClassificationModel(
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
            loss_fn=_neg_f1_loss_fn,
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            num_features=num_features,
        )
        self._lam = lam


def _pos_f1_loss_fn(targets: np.ndarray, preds: np.ndarray) -> float:
    p1_ = np.mean(targets)
    p_1 = np.mean(preds)
    p11 = np.mean(targets * preds)
    p1sum = p1_ + p_1
    return -p11 / p1sum if p1sum != 0 else np.nan


def _neg_f1_loss_fn(targets: np.ndarray, preds: np.ndarray) -> float:
    p0_ = 1 - np.mean(targets)
    p_0 = 1 - np.mean(preds)
    p00 = np.mean((1 - targets) * (1 - preds))
    p0sum = p0_ + p_0
    return -p00 / p0sum if p0sum != 0 else np.nan
