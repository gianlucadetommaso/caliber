from functools import partial
from typing import Optional

from caliber.binary_classification.metrics.focal_loss import focal_loss
from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.mixins.fit.smooth_fit import (
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
        has_bivariate_slope: bool = False,
        num_features: int = 0,
        gamma: float = 2.0,
    ):
        super().__init__(
            loss_fn=partial(focal_loss, gamma=gamma),
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            has_bivariate_slope=has_bivariate_slope,
            num_features=num_features,
        )
