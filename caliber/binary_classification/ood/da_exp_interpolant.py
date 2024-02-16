from typing import Any, Optional

import numpy as np
from scipy import stats


class DistanceAwareInterpolantBinaryClassificationModel:
    def __init__(self, model: Optional[Any] = None, conf_distance: float = 0.99):
        self.model = model
        # self._train_cdf = None
        self._rv = None
        self.conf_distance = conf_distance
        self._quantile_distance = None
        # self._max_distance = None
        # self._cdf = None

    def fit(self, probs: np.ndarray, distances: np.ndarray, targets: np.ndarray):
        if self.model is not None:
            self.model.fit(probs, targets)
        self._quantile_distance = np.quantile(distances, self.conf_distance)
        # self._max_distance = np.max(distances)
        # self._cdf = expon(0, self._max_distance).cdf
        # self._train_cdf = ecdf(distances).cdf.evaluate
        self._rv = stats.expon(*stats.expon.fit(distances))
        # self._train_distances = distances

    def predict_proba(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        probs = np.copy(probs)
        if self.model is not None:
            probs = self.model.predict_proba(probs)
        # cdf = self._cdf(distances - self._quantile_distance)
        # return (1 - cdf) * probs + 0.5 * cdf
        # cdf = ecdf(distances).cdf.evaluate
        # ood_weights = np.abs(cdf(distances) - self._train_cdf(distances))
        # ood_weights = np.clip(self._rv.cdf(distances) - self._rv.interval(0.90)[1], 0, np.inf)
        # ood_weights = ood_weights * (ood_weights > 0.95)
        # return (1 - ood_weights) * probs + 0.5 * ood_weights
        # ks = ks_2samp(self._train_distances, distances)
        # cdf = expon(0, ks.statistic_location).cdf(distances - ks.statistic_location)
        # ood_weight = self._rv.cdf(distances)
        # return (1 - ood_weight) * probs + 0.5 * ood_weight
        # conf = 0.5
        # right = 1 - 0.5 * (1 - conf)
        # left = 0.5 * (1 - conf)
        # low, high = self._rv.interval(conf)
        # cdfs = self._rv.cdf(distances)
        # ood_weight = ((distances >= high) * (cdfs - right) + (distances <= low) * (0.05 - left)) / 0.05
        # return (1 - ood_weight) * probs + 0.5 * ood_weight
        cdf = self._get_cdf(distances)
        return (1 - cdf) * probs + 0.5 * cdf

    def predict(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs, distances) >= 0.5).astype(int)

    def _get_cdf(self, distances: np.ndarray) -> np.ndarray:
        d = distances - self._quantile_distance
        return np.where(d > 0, self._rv.cdf(d), 0.0)
