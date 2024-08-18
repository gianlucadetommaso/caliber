from functools import partial
from typing import Optional

import numpy as np

from caliber.binary_classification.metrics.asce import (
    average_smooth_squared_calibration_error,
)
from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.mixins.fit.smooth_fit import (
    LinearScalingSmoothFitBinaryClassificationMixin,
)


class SmoothLinearScalingBinaryClassificationModel(
    LinearScalingSmoothFitBinaryClassificationMixin,
    CalibrationLinearScalingBinaryClassificationModel,
):
    def __init__(
        self,
        lam: float = 0.01,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        sigma: float = 0.1,
        n_bins: int = 10,
    ) -> None:
        super().__init__(
            loss_fn=partial(
                average_smooth_squared_calibration_error, n_bins=n_bins, sigma=sigma
            ),
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            has_bivariate_slope=False,
        )
        self._lam = lam
        self._sigma = sigma
