from typing import Optional

from caliber.binary_classification.metrics.ece import expected_calibration_error
from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.mixins.fit.brute_fit import (
    LinearScalingBruteFitBinaryClassificationMixin,
)


class ECELinearScalingBinaryClassificationModel(
    LinearScalingBruteFitBinaryClassificationMixin,
    CalibrationLinearScalingBinaryClassificationModel,
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        lam: float = 0.01,
        has_intercept: bool = True,
        has_bivariate_slope: bool = False,
        num_features: int = 0,
    ):
        super().__init__(
            loss_fn=expected_calibration_error,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            has_bivariate_slope=has_bivariate_slope,
            num_features=num_features,
        )
        self._lam = lam
