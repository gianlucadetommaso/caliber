from typing import Optional

import numpy as np

from caliber.binary_classification.linear_scaling.performance.base import (
    CustomPerformanceBinaryClassificationLinearScaling,
)


class BalancedAccuracyBinaryClassificationLinearScaling(
    CustomPerformanceBinaryClassificationLinearScaling
):
    def __init__(
        self,
        threshold: float,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
    ):
        super().__init__(
            loss_fn=balanced_accuracy_loss_fn,
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )


def balanced_accuracy_loss_fn(preds: np.ndarray, targets: np.ndarray) -> float:
    n_pos_targets = np.sum(targets)
    n_neg_targets = len(targets) - n_pos_targets
    tpr = np.sum(targets * preds) / n_pos_targets if n_pos_targets > 0 else 0.0
    tnr = (
        np.sum((1 - targets) * (1 - preds)) / n_neg_targets
        if n_neg_targets > 0
        else 0.0
    )
    return -(tpr + tnr)
