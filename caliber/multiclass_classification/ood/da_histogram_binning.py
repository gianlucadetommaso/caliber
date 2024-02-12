import numpy as np
from scipy.stats import expon


class DistanceAwareHistogramBinningMulticlassClassificationModel:
    def __init__(
        self,
        n_prob_bins: int = 10,
        n_dist_bins: int = 10,
        conf_distance: float = 0.95,
        min_prob_bin: float = 0.01,
    ):
        self.n_prob_bins = n_prob_bins
        self.n_dist_bins = n_dist_bins
        self.conf_distance = conf_distance
        self._min_prob_bin = min_prob_bin
        self._params = None
        self._quantile_distance = None
        self._prob_bin_edges = None
        self._dist_bin_edges = None
        self._ood_threshold = None
        self._max_distance = None
        self._n_classes = None

    def fit(self, probs: np.ndarray, distances: np.ndarray, targets: np.ndarray):
        self._quantile_distance = np.quantile(distances, self.conf_distance)
        self._max_distance = np.max(distances)
        self._ood_threshold = expon(0, self._max_distance)
        self._n_classes = probs.shape[1]

        self._prob_bin_edges = self._get_prob_bin_edges()
        self._dist_bin_edges = self._get_dist_bin_edges()

        prob_bin_indices = np.digitize(probs, self._prob_bin_edges)
        dist_bin_indices = np.digitize(distances, self._dist_bin_edges)
        top_class_indices = np.argmax(probs, axis=1)

        self._params = np.empty(
            (self.n_prob_bins + 1, self.n_dist_bins + 1, self._n_classes)
        )

        for i in range(1, self.n_prob_bins + 2):
            for j in range(1, self.n_dist_bins + 2):
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

        for i in range(1, self.n_prob_bins + 2):
            for j in range(1, self.n_dist_bins + 2):
                for c in range(0, self._n_classes):
                    mask = self._get_mask(
                        i, j, c, prob_bin_indices, dist_bin_indices, top_class_indices
                    )
                    if not np.isnan(self._params[i - 1, j - 1, c]):
                        probs[mask, c] = self._params[i - 1, j - 1, c]
                    else:
                        cdf = self._get_cdf(distances[mask])
                        probs[mask, c] = (1 - cdf) * probs[
                            mask, c
                        ] + cdf / self._n_classes
        return probs

    def predict(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(probs, distances), axis=1)

    def _get_prob_bin_edges(self) -> np.ndarray:
        return np.linspace(0, 1, self.n_prob_bins + 1)

    def _get_dist_bin_edges(self) -> np.ndarray:
        return np.linspace(0, self._max_distance, self.n_dist_bins + 1)

    def _fit_bin(self, i: int, j: int, c: int, mask: np.ndarray, targets: np.ndarray):
        prob_bin = np.mean(mask)
        if prob_bin >= self._min_prob_bin:
            cdf = self._ood_threshold.cdf(
                self._dist_bin_edges[j - 1] - self._quantile_distance
            )
            self._params[i - 1, j - 1, c] = (1 - cdf) * np.mean(
                targets[mask] == c
            ) + 0.5 * cdf
        else:
            self._params[i - 1, j - 1, c] = np.nan

    @staticmethod
    def _get_mask(
        prob_bin_idx: int,
        dist_bin_idx: int,
        class_idx: int,
        prob_bin_indices: np.ndarray,
        dist_bin_indices: np.ndarray,
        top_class_indices: np.ndarray,
    ) -> np.ndarray:
        return (
            (prob_bin_indices[:, class_idx] == prob_bin_idx)
            * (top_class_indices == class_idx)
            * (dist_bin_indices == dist_bin_idx)
        )

    def _get_cdf(self, d: float) -> np.ndarray:
        d -= self._quantile_distance
        return self._ood_threshold.cdf(d) * (d > 0)
