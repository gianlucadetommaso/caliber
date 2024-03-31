from functools import partial
from typing import Optional

from caliber.binary_classification.metrics.focal_loss import focal_loss
from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.linear_scaling_smooth_fit_mixin import (
    LinearScalingSmoothFitBinaryClassificationMixin,
)


class FocalLinearScalingBinaryClassificationModel(
    LinearScalingSmoothFitBinaryClassificationMixin,
    CalibrationLinearScalingBinaryClassificationModel,
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        gamma: float = 2.0,
    ):
        super().__init__(
            loss_fn=partial(focal_loss, gamma=gamma),
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )
