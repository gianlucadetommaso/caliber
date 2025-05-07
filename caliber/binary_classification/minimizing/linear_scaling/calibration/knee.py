from functools import partial
from typing import Optional

from caliber.binary_classification.metrics.knee_distance import knee_point_distance
from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.mixins.fit.brute_fit import (
    LinearScalingBruteFitBinaryClassificationMixin,
)


class KneePointLinearScalingBinaryClassificationModel(
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
        n_thresholds: int = 100,
    ):
        super().__init__(
            loss_fn=partial(knee_point_distance, n_thresholds=n_thresholds),
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            has_bivariate_slope=has_bivariate_slope,
            num_features=num_features,
        )
        self._lam = lam
