from typing import Callable, Optional

import numpy as np

from caliber.binary_classification.linear_scaling.base import (
    CustomBinaryClassificationLinearScaling,
)
from caliber.binary_classification.linear_scaling.brute_fit_mixin_linear_scaling import (
    BinaryClassificationLinearScalingBruteFitMixin,
)


class CustomPerformanceBinaryClassificationLinearScaling(
    BinaryClassificationLinearScalingBruteFitMixin,
    CustomBinaryClassificationLinearScaling,
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        threshold: float,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
    ):
        super().__init__(loss_fn, minimize_options, has_intercept)
        self._threshold = threshold

    def _get_output_for_loss(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return (self._predict_proba(params, probs) >= self._threshold).astype(int)

    @property
    def threshold(self):
        return self._threshold

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs) >= self._threshold).astype(int)
