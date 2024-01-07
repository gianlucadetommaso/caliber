from typing import Optional

from caliber.binary_classification.linear_scaling.brute_fit_mixin_linear_scaling import (
    BinaryClassificationLinearScalingBruteFitMixin,
)
from caliber.binary_classification.linear_scaling.calibration.base import (
    CustomCalibrationBinaryClassificationLinearScaling,
)
from caliber.binary_classification.metrics.ece import expected_calibration_error


class ECEBinaryClassificationLinearScaling(
    BinaryClassificationLinearScalingBruteFitMixin,
    CustomCalibrationBinaryClassificationLinearScaling,
):
    def __init__(
        self, minimize_options: Optional[dict] = None, has_intercept: bool = True
    ):
        super().__init__(
            loss_fn=expected_calibration_error,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )
