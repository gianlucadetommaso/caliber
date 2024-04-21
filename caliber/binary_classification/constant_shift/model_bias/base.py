import numpy as np

from caliber.binary_classification.base import AbstractBinaryClassificationModel
from caliber.binary_classification.metrics.bias import model_bias
from caliber.binary_classification.pred_from_probs_mixin import (
    PredFromProbsBinaryClassificationMixin,
)


class ModelBiasConstantShiftBinaryClassificationModel(
    PredFromProbsBinaryClassificationMixin, AbstractBinaryClassificationModel
):
    def __init__(self, step_size: float = 1.0):
        super().__init__(threshold=0.5)
        self.step_size = step_size

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> None:
        self._params = model_bias(targets, probs)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return np.clip(probs + self.step_size * self._params, 0, 1)
