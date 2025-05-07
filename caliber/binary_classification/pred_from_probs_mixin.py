import numpy as np

from caliber.binary_classification.checks_mixin import BinaryClassificationChecksMixin


class PredFromProbsBinaryClassificationMixin(BinaryClassificationChecksMixin):
    def __init__(self, threshold: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def predict(self, probs: np.ndarray) -> np.ndarray:
        self._check_probs(probs)
        probs = self.predict_proba(probs)
        return (probs > self.threshold).astype(int)
