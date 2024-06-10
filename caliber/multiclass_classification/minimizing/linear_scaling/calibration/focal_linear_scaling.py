from functools import partial
from typing import Optional

from caliber.multiclass_classification.metrics import focal_loss
from caliber.multiclass_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingMulticlassClassificationModel,
)
from caliber.multiclass_classification.minimizing.linear_scaling.linear_scaling_smooth_fit_mixin_ import (
    LinearScalingSmoothFitMulticlassClassificationMixin,
)


class FocalLinearScalingMulticlassClassificationModel(
    LinearScalingSmoothFitMulticlassClassificationMixin,
    CalibrationLinearScalingMulticlassClassificationModel,
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        has_shared_intercept: bool = False,
        has_cross_slopes: bool = True,
        has_shared_slope: bool = False,
        gamma: float = 2.0,
    ):
        super().__init__(
            partial(focal_loss, gamma=gamma),
            minimize_options,
            has_intercept=has_intercept,
            has_shared_intercept=has_shared_intercept,
            has_shared_slope=has_shared_slope,
            has_cross_slopes=has_cross_slopes,
        )
