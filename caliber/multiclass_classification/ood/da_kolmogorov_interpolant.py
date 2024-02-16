from typing import Any, Optional

import numpy as np
from scipy import stats
from scipy.special import kolmogorov


class DistanceAwareKolmogorovInterpolantMulticlassClassificationModel:
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
        )[:, None]
        return w * probs + (1 - w) / probs.shape[1]

    def predict(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(probs, distances), axis=1)
