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
    ):
        super().__init__(
            loss_fn=loss_fn,
            threshold=0.5,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )

    def _get_output_for_loss(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self._predict_proba(params, probs)
