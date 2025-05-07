from typing import Optional

import numpy as np

from caliber.binary_classification.metrics.precrec_error import precision_recall_error
from caliber.binary_classification.minimizing.linear_scaling.calibration.base import (
    CalibrationLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.minimizing.linear_scaling.mixins.fit.brute_fit import (
    LinearScalingBruteFitBinaryClassificationMixin,
)
from caliber.binary_classification.utils.knee_point import knee_point


class PrecisionRecallLinearScalingBinaryClassificationModel(
    LinearScalingBruteFitBinaryClassificationMixin,
    CalibrationLinearScalingBinaryClassificationModel,
):
    def __init__(
        self,
        threshold: Optional[float] = None,
        lam: float = 0.01,
        minimize_options: Optional[dict] = None,
        has_intercept: bool = True,
        num_features: int = 0,
        n_thresholds: int = 100,
    ):
        super().__init__(
            loss_fn=self._precision_recall_error,
            minimize_options=minimize_options,
            has_intercept=has_intercept,
            has_bivariate_slope=False,
            num_features=num_features,
        )
        self._lam = lam
        if threshold is None:
            self._tune_threshold = True
        else:
            self._tune_threshold = True
            self.threshold = threshold
        self._n_thresholds = n_thresholds

    def _precision_recall_error(self, targets: np.ndarray, probs: np.ndarray) -> float:
        self._maybe_update_threshold(probs, targets)
        return precision_recall_error(targets, probs, self.threshold)

    def _maybe_update_threshold(self, probs: np.ndarray, targets: np.ndarray) -> None:
        if self._tune_threshold:
            self.threshold = knee_point(probs, targets, self._n_thresholds)[2]
