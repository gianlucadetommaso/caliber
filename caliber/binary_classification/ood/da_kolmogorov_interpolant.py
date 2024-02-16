from typing import Any, Optional

import numpy as np
from scipy import stats
from scipy.special import kolmogorov


class DistanceAwareKolmogorovInterpolantBinaryClassificationModel:
    def __init__(self, model: Optional[Any] = None):
        self.model = model
        self._train_ecdf = None

    def fit(self, probs: np.ndarray, distances: np.ndarray, targets: np.ndarray):
        if self.model is not None:
            self.model.fit(probs, targets)
        self._train_ecdf = stats.ecdf(distances).cdf

    def predict_proba(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        probs = np.copy(probs)
        if self.model is not None:
            probs = self.model.predict_proba(probs)

        ecdf = stats.ecdf(distances).cdf
        w = kolmogorov(
            np.sqrt(len(distances))
            * np.abs(ecdf.evaluate(distances) - self._train_ecdf.evaluate(distances))
        )
        return w * probs + 0.5 * (1 - w)

    def predict(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs, distances) >= 0.5).astype(int)
