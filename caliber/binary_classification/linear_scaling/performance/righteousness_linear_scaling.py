from typing import Optional

import numpy as np

from caliber.binary_classification.linear_scaling.performance.base import (
    CustomPerformanceBinaryClassificationLinearScaling,
)


class RighteousnessBinaryClassificationLinearScaling(
    CustomPerformanceBinaryClassificationLinearScaling
):
    def __init__(
        self,
        threshold: float,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
    ):
        super().__init__(
            loss_fn=righteousness_loss_fn,
            threshold=threshold,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
        )


def righteousness_loss_fn(preds: np.ndarray, targets: np.ndarray) -> float:
    p11 = np.sum(targets * preds)
    p00 = np.sum((1 - targets) * (1 - preds))
    p01 = np.sum((1 - targets) * preds)
    p10 = np.sum(targets * (1 - preds))
    return -(1 - p01 * p10) * p00 * p11
