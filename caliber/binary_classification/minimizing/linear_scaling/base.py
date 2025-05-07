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
        has_bivariate_slope: bool = False,
        num_features: int = 0,
    ):
        self._has_intercept = has_intercept
        self._has_bivariate_slope = has_bivariate_slope
        super().__init__(
            loss_fn, threshold, minimize_options, num_features=num_features
        )

    def _predict_proba(
        self,
        params: np.ndarray,
        probs: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        if self._has_intercept:
            if self._has_bivariate_slope:
                logits = (
                    params[0]
                    + params[1] * np.log(probs)
                    - params[2] * np.log(1 - probs)
                )
                if features is not None:
                    logits += features @ params[3:]
                return expit(logits)

            logits = params[0] + params[1] * logit(probs)
            if features is not None:
                logits += features @ params[2:]
            return expit(logits)

        if self._has_bivariate_slope:
            logits = params[0] * np.log(probs) - params[1] * np.log(1 - probs)
            if features is not None:
                logits += features @ params[2:]
            return expit(logits)

        logits = logit(probs)
        if features is not None:
            logits += features @ params[1:]
        return expit(params * logit(probs))
