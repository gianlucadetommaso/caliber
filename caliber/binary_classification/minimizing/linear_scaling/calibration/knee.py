from typing import Optional

import numpy as np

from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.linear_scaling_brute_fit_mixin import (
    LinearScalingBruteFitBinaryClassificationMixin,
)


class KneePointLinearScalingBinaryClassificationModel(
    LinearScalingBruteFitBinaryClassificationMixin,
    CalibrationLinearScalingBinaryClassificationModel,
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        has_bivariate_slope: bool = False,
    ):
        super().__init__(
            loss_fn=_knee_point_distance,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            has_bivariate_slope=has_bivariate_slope,
        )


def _knee_point_distance(
    targets: np.ndarray, probs: np.ndarray, n_thresholds: int = 10
) -> float:
    thresholds = np.linspace(0, 1, n_thresholds)
    preds = probs[:, None] >= thresholds[None]
    n_pos_targets = np.sum(targets)
    n_pos_preds = np.sum(preds, 0)
    joint = np.sum(targets[:, None] * preds, 0)
    recall = np.where(n_pos_targets > 0, joint / n_pos_targets, 0.0)
    precision = np.where(n_pos_preds > 0, joint / n_pos_preds, 0.0)

    idx = np.argmax(precision + recall)
    return (1 - precision[idx]) ** 2 + (1 - recall[idx]) ** 2
