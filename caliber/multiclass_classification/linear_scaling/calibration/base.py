import abc
from typing import Callable, Optional

import numpy as np

from caliber.multiclass_classification.linear_scaling.base import (
    CustomMulticlassClassificationLinearScaling,
)


class CustomCalibrationMulticlassClassificationLinearScaling(
    CustomMulticlassClassificationLinearScaling, abc.ABC
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        minimize_options: Optional[dict] = None,
        has_shared_slope: bool = True,
    ):
        super().__init__(loss_fn, minimize_options, has_shared_slope)

    def _get_output_for_loss(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self._predict_proba(params, probs)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(probs), 1)
