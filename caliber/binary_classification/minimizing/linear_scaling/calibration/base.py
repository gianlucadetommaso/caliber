import abc
from typing import Callable, Optional

import numpy as np

from caliber.binary_classification.minimizing.linear_scaling.base import (
    LinearScalingBinaryClassificationModel,
)


class CalibrationLinearScalingBinaryClassificationModel(
    LinearScalingBinaryClassificationModel, abc.ABC
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        has_bivariate_slope: bool = False,
        num_features: int = 0,
    ):
        super().__init__(
            loss_fn=loss_fn,
            threshold=0.5,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            has_bivariate_slope=has_bivariate_slope,
            num_features=num_features,
        )

    def _get_output_for_loss(
        self,
        params: np.ndarray,
        probs: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._predict_proba(params, probs, features)
