from typing import Optional

from caliber.multiclass_classification.metrics import brier_score_loss
from caliber.multiclass_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingMulticlassClassificationModel,
)
from caliber.multiclass_classification.minimizing.linear_scaling.linear_scaling_smooth_fit_mixin_ import (
    LinearScalingSmoothFitMulticlassClassificationMixin,
)


class BrierLinearScalingMulticlassClassificationModel(
    LinearScalingSmoothFitMulticlassClassificationMixin,
    CalibrationLinearScalingMulticlassClassificationModel,
):
    def __init__(
        self, minimize_options: Optional[dict] = None, has_shared_slope: bool = True
    ):
        super().__init__(
            loss_fn=brier_score_loss,
            minimize_options=minimize_options,
            has_shared_slope=has_shared_slope,
        )
