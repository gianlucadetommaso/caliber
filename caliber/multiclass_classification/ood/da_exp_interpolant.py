from typing import Any, Optional

import numpy as np
from scipy import stats


class DistanceAwareExponentialInterpolantMulticlassClassificationModel:
    def __init__(self, model: Optional[Any] = None, conf_distance: float = 0.99):
        self.model = model
        self.conf_distance = conf_distance
        self._quantile_distance = None
        self._rv = None

    def fit(self, probs: np.ndarray, distances: np.ndarray, targets: np.ndarray):
        if self.model is not None:
            self.model.fit(probs, targets)
        self._quantile_distance = np.quantile(distances, self.conf_distance)
        self._rv = stats.expon(*stats.expon.fit(distances))

    def predict_proba(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        probs = np.copy(probs)
        if self.model is not None:
            probs = self.model.predict_proba(probs)
        cdf = self._get_cdf(distances)[:, None]
        return (1 - cdf) * probs + cdf / probs.shape[1]

    def predict(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(probs, distances), axis=1)

    def _get_cdf(self, distances: np.ndarray) -> np.ndarray:
        d = distances - self._quantile_distance
        return np.where(d > 0, self._rv.cdf(d), 0.0)
