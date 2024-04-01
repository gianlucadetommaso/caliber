from typing import Optional

from caliber.binary_classification.metrics.asce import average_squared_calibration_error
from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.linear_scaling_brute_fit_mixin import (
    LinearScalingBruteFitBinaryClassificationMixin,
)


class ASCELinearScalingBinaryClassificationModel(
    LinearScalingBruteFitBinaryClassificationMixin,
    CalibrationLinearScalingBinaryClassificationModel,
):
    def __init__(
        self, minimize_options: Optional[dict] = None, has_intercept: bool = True
    ):
        super().__init__(
            loss_fn=average_squared_calibration_error,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )
