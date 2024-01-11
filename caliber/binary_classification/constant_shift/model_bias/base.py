import numpy as np

from caliber.binary_classification.metrics.bias import model_bias


class ModelBiasBinaryClassificationConstantShift:
    def __init__(self, step_size: float = 1.0):
        self.step_size = step_size
        self._params = None

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> None:
        self._params = model_bias(targets, probs)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return np.clip(probs + self.step_size * self._params, 0, 1)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs) >= 0.5).astype(int)
