from typing import Optional

from caliber.multiclass_classification.linear_scaling.calibration.base import (
    CustomCalibrationMulticlassClassificationLinearScaling,
)
from caliber.multiclass_classification.linear_scaling.smooth_fit_mixin_linear_scaling import (
    MulticlassClassificationLinearScalingSmoothFitMixin,
)
from caliber.multiclass_classification.metrics import brier_score_loss


class BrierMulticlassClassificationLinearScaling(
    MulticlassClassificationLinearScalingSmoothFitMixin,
    CustomCalibrationMulticlassClassificationLinearScaling,
):
    def __init__(
        self, minimize_options: Optional[dict] = None, has_shared_slope: bool = True
    ):
        super().__init__(
            loss_fn=brier_score_loss,
            minimize_options=minimize_options,
            has_shared_slope=has_shared_slope,
        )
