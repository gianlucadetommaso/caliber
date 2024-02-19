from functools import partial
from typing import Optional

from caliber.multiclass_classification.linear_scaling.calibration.base import (
    CustomCalibrationMulticlassClassificationLinearScaling,
)
from caliber.multiclass_classification.linear_scaling.smooth_fit_mixin_linear_scaling import (
    MulticlassClassificationLinearScalingSmoothFitMixin,
)
from caliber.multiclass_classification.metrics import focal_loss


class FocalMulticlassClassificationLinearScaling(
    MulticlassClassificationLinearScalingSmoothFitMixin,
    CustomCalibrationMulticlassClassificationLinearScaling,
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        has_shared_slope: bool = False,
        gamma: float = 2.0,
    ):
        super().__init__(
            loss_fn=partial(focal_loss, gamma=gamma),
            minimize_options=minimize_options,
            has_shared_slope=has_shared_slope,
        )
