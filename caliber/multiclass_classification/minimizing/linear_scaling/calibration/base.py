import abc
from typing import Callable, Optional

import numpy as np

from caliber.multiclass_classification.minimizing.linear_scaling.base import (
    LinearScalingMulticlassClassificationModel,
)


class CalibrationLinearScalingMulticlassClassificationModel(
    LinearScalingMulticlassClassificationModel, abc.ABC
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        has_shared_intercept: bool = False,
        has_cross_slopes: bool = True,
        has_shared_slope: bool = False,
    ):
        super().__init__(
            loss_fn,
            minimize_options,
            has_intercept=has_intercept,
            has_shared_intercept=has_shared_intercept,
            has_shared_slope=has_shared_slope,
            has_cross_slopes=has_cross_slopes,
        )

    def _get_output_for_loss(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self._predict_proba(params, probs)
