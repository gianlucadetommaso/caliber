import abc
from typing import Callable, Optional

import numpy as np

from caliber.binary_classification.linear_scaling.base import (
    CustomBinaryClassificationLinearScaling,
)


class CustomCalibrationBinaryClassificationLinearScaling(
    CustomBinaryClassificationLinearScaling, abc.ABC
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
    ):
        super().__init__(loss_fn, minimize_options, has_intercept)

    def _get_output_for_loss(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self._predict_proba(params, probs)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs) >= 0.5).astype(int)
