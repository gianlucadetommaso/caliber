from typing import Optional

from sklearn.metrics import brier_score_loss

from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.linear_scaling_smooth_fit_mixin import (
    LinearScalingSmoothFitBinaryClassificationMixin,
)


class BrierLinearScalingBinaryClassificationModel(
    LinearScalingSmoothFitBinaryClassificationMixin,
    CalibrationLinearScalingBinaryClassificationModel,
):
    def __init__(
        self, minimize_options: Optional[dict] = None, has_intercept: bool = True
    ):
        super().__init__(
            loss_fn=brier_score_loss,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )
