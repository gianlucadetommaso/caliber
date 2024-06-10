import abc
from typing import Callable, Optional

import numpy as np
from scipy.special import softmax

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
        has_intercept: bool = True,
        has_shared_intercept: bool = False,
        has_shared_slope: bool = False,
        has_cross_slopes: bool = True,
    ):
        if has_cross_slopes and has_shared_slope:
            raise ValueError(
                "Only one between `has_shared_slope` and `has_cross_slope` can be set to `True`."
            )
        self._has_intercept = has_intercept
        self._has_shared_intercept = has_shared_intercept
        self._has_shared_slope = has_shared_slope
        self._has_cross_slopes = has_cross_slopes
        super().__init__(loss_fn, minimize_options)

    def _predict_proba(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 1e-6, np.inf)
        if self._has_intercept:
            if self._has_shared_intercept:
                intercept, slope = params[0], params[1:]
            else:
                intercept, slope = params[: self._n_classes], params[self._n_classes :]
        else:
            intercept, slope = 0.0, params
        if self._has_cross_slopes:
            slope = slope.reshape(self._n_classes, -1)
            return softmax(intercept + np.matmul(np.log(probs), slope), axis=1)
        return softmax(intercept + slope * np.log(probs), axis=1)
