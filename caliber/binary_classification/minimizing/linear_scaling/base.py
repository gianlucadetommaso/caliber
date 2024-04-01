import abc
from typing import Callable, Optional

import numpy as np
from scipy.special import expit, logit

from caliber.binary_classification.minimizing.base import (
    MinimizingBinaryClassificationModel,
)


class LinearScalingBinaryClassificationModel(
    MinimizingBinaryClassificationModel, abc.ABC
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        threshold: float,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
    ):
        self._has_intercept = has_intercept
        super().__init__(loss_fn, threshold, minimize_options)

    def _predict_proba(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return (
            expit(params[0] + params[1] * logit(probs))
            if self._has_intercept
            else expit(params * logit(probs))
        )
