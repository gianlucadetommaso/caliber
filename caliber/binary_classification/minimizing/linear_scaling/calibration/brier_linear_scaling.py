from typing import Optional

from sklearn.metrics import brier_score_loss

from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.mixins.fit.smooth_fit import (
    LinearScalingSmoothFitBinaryClassificationMixin,
)


class BrierLinearScalingBinaryClassificationModel(
    LinearScalingSmoothFitBinaryClassificationMixin,
    CalibrationLinearScalingBinaryClassificationModel,
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        has_bivariate_slope: bool = False,
        num_features: int = 0,
    ):
        super().__init__(
            loss_fn=brier_score_loss,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            has_bivariate_slope=has_bivariate_slope,
            num_features=num_features,
        )
