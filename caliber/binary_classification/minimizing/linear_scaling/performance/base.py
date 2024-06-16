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
    ):
        super().__init__(
            loss_fn,
            threshold,
            minimize_options,
            has_intercept,
            has_bivariate_slope=False,
        )

    def _get_output_for_loss(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return (self._predict_proba(params, probs) >= self.threshold).astype(int)
