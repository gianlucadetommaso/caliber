from typing import Optional

from sklearn.metrics import log_loss

from caliber.binary_classification.linear_scaling.calibration.base import (
    CustomCalibrationBinaryClassificationLinearScaling,
)
from caliber.binary_classification.linear_scaling.smooth_fit_mixin_linear_scaling import (
    BinaryClassificationLinearScalingSmoothFitMixin,
)


class CrossEntropyBinaryClassificationLinearScaling(
    BinaryClassificationLinearScalingSmoothFitMixin,
    CustomCalibrationBinaryClassificationLinearScaling,
):
    def __init__(
        self, minimize_options: Optional[dict] = None, has_intercept: bool = True
    ):
        super().__init__(
            loss_fn=log_loss,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )
