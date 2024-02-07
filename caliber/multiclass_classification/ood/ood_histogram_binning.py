import numpy as np
from scipy.stats import expon


class OODHistogramBinningMulticlassClassificationModel:
    def __init__(
        self,
        n_prob_bins: int = 10,
        n_dist_bins: int = 10,
        conf_distance: float = 0.95,
        conf_ood_threshold: float = 0.99,
    ):
        self._n_prob_bins = n_prob_bins
        self._n_dist_bins = n_dist_bins
        self._params = None
        self._prob_bin_edges = None
        self._dist_bin_edges = None
        self._ood_threshold = None
        self._conf_distance = conf_distance
        self._conf_ood_threshold = conf_ood_threshold
        self._n_classes = None

    def fit(self, probs: np.ndarray, distances: np.ndarray, targets: np.ndarray):
        self._conf_distance = np.quantile(distances, self._conf_distance)
        self._ood_threshold = expon(0, self._conf_distance)
        self._n_classes = probs.shape[1]

        self._prob_bin_edges = self._get_prob_bin_edges()
        self._dist_bin_edges = self._get_dist_bin_edges()

        prob_bin_indices = np.digitize(probs, self._prob_bin_edges)
        dist_bin_indices = np.digitize(distances, self._dist_bin_edges)
        top_class_indices = np.argmax(probs, axis=1)

        self._params = 0.5 * np.ones(
            (self._n_prob_bins + 1, self._n_dist_bins + 1, self._n_classes)
        )

        for i in range(1, self._n_prob_bins + 2):
            for j in range(1, self._n_dist_bins + 2):
                for c in range(self._n_classes):
                    mask = self._get_mask(
                        i, j, c, prob_bin_indices, dist_bin_indices, top_class_indices
                    )
                    self._fit_bin(i, j, c, mask, targets)

    def predict_proba(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        if self._prob_bin_edges is None:
            raise ValueError("Run `fit` first.")
        if probs.shape[1] != self._n_classes:
            raise ValueError(
                "The number of classes when fitting and predicting must be the same."
            )

        prob_bin_indices = np.digitize(probs, self._prob_bin_edges)
        dist_bin_indices = np.digitize(distances, self._dist_bin_edges)
        top_class_indices = np.argmax(probs, axis=1)

        probs = np.copy(probs)

        for i in range(1, self._n_prob_bins + 2):
            for j in range(1, self._n_dist_bins + 2):
                for c in range(0, self._n_classes):
                    mask = self._get_mask(
                        i, j, c, prob_bin_indices, dist_bin_indices, top_class_indices
                    )
                    probs[mask, c] = self._params[i - 1, j - 1, c]
        return probs

    def predict(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(probs, distances), axis=1)

    def _get_prob_bin_edges(self):
        return np.linspace(0, 1, self._n_prob_bins + 1)

    def _get_dist_bin_edges(self):
        b = self._ood_threshold.interval(self._conf_ood_threshold)[1]
        h = (b - self._conf_distance) / self._n_dist_bins
        return np.linspace(self._conf_distance + h, b, self._n_dist_bins)

    def _fit_bin(self, i: int, j: int, c: int, mask: np.ndarray, targets: np.ndarray):
        prob_bin = np.mean(mask)
        if prob_bin != 0:
            cdf = self._ood_threshold.cdf(
                self._dist_bin_edges[j - 1] - self._conf_distance
            )
            self._params[i - 1, j - 1, c] = (1 - cdf) * np.mean(
                targets[mask] == c
            ) + 0.5 * cdf

    def _get_mask(
        self,
        prob_bin_idx: int,
        dist_bin_idx: int,
        class_idx: int,
        prob_bin_indices: np.ndarray,
        dist_bin_indices: np.ndarray,
        top_class_indices: np.ndarray,
    ) -> np.ndarray:
        cond = (prob_bin_indices[:, class_idx] == prob_bin_idx) * (
            top_class_indices == class_idx
        )
        if dist_bin_idx > self._conf_distance:
            cond *= dist_bin_indices == dist_bin_idx
        return cond
