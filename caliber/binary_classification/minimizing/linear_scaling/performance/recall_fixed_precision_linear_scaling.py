from functools import partial
from typing import Optional

import numpy as np

from caliber.binary_classification.minimizing.linear_scaling.performance.base import (
    PerformanceLinearScalingBinaryClassificationModel,
)


class RecallFixedPrecisionLinearScalingBinaryClassificationModel(
    PerformanceLinearScalingBinaryClassificationModel
):
    def __init__(
        self,
        threshold: float,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        min_precision: float = 0.8,
    ):
        super().__init__(
            loss_fn=partial(_recall_fixed_precision, min_precision=min_precision),
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )


def _recall_fixed_precision(
    targets: np.ndarray, preds: np.ndarray, min_precision: float
) -> float:
    n_pos_targets = np.sum(targets)
    n_pos_preds = np.sum(preds)
    joint = np.sum(targets * preds)
    recall = joint / n_pos_targets if n_pos_targets > 0 else 0.0
    precision = joint / n_pos_preds if n_pos_preds > 0 else 0.0
    cond = precision >= min_precision
    return -recall * cond