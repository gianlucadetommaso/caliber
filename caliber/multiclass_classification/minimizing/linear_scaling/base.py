import abc
from typing import Callable, Optional

import numpy as np
from scipy.special import logit, softmax

from caliber.multiclass_classification.minimizing.base import (
    MinimizingMulticlassClassificationModel,
)


class LinearScalingMulticlassClassificationModel(
    MinimizingMulticlassClassificationModel, abc.ABC
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        minimize_options: Optional[dict] = None,
        has_shared_slope: bool = True,
    ):
        self._has_shared_slope = has_shared_slope
        super().__init__(loss_fn, minimize_options)

    @staticmethod
    def _predict_proba(params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return softmax(params * logit(probs), axis=1)