from typing import Callable, Optional

import numpy as np

from caliber.binary_classification.minimizing.linear_scaling.base import (
    LinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.mixins.fit.brute_fit import (
    LinearScalingBruteFitBinaryClassificationMixin,
)


class PerformanceLinearScalingBinaryClassificationModel(
    LinearScalingBruteFitBinaryClassificationMixin,
    LinearScalingBinaryClassificationModel,
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        threshold: float,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        num_features: int = 0,
    ):
        super().__init__(
            loss_fn,
            threshold,
            minimize_options,
            has_intercept,
            has_bivariate_slope=False,
            num_features=num_features,
        )

    def _get_output_for_loss(
        self,
        params: np.ndarray,
        probs: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return (self._predict_proba(params, probs, features) >= self.threshold).astype(
            int
        )
