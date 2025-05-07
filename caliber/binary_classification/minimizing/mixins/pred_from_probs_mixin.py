from typing import Optional

import numpy as np

from caliber.binary_classification.checks_mixin import BinaryClassificationChecksMixin


class PredFromProbsBinaryClassificationMixin(BinaryClassificationChecksMixin):
    def __init__(self, threshold: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def predict(
        self, probs: np.ndarray, features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self._check_probs(probs)
        self._check_features(features)
        probs = self._predict_proba(self._params, probs, features)
        return (probs > self.threshold).astype(int)
