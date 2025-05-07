from typing import Optional

import numpy as np

from caliber.binary_classification.minimizing.linear_scaling.performance.base import (
    PerformanceLinearScalingBinaryClassificationModel,
)


class PositiveNegativeRatesLinearScalingBinaryClassificationModel(
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
            loss_fn=_true_positive_negative_rates_loss_fn,
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            num_features=num_features,
        )
        self._lam = lam


def _true_positive_negative_rates_loss_fn(
    targets: np.ndarray, preds: np.ndarray
) -> float:
    n_pos_targets = np.sum(targets)
    n_neg_targets = len(targets) - n_pos_targets
    tpr = np.sum(targets * preds) / n_pos_targets if n_pos_targets > 0 else 0
    tnr = (
        np.sum((1 - targets) * (1 - preds)) / n_neg_targets if n_neg_targets > 0 else 0
    )
    return -tpr * tnr / (tpr + tnr)
