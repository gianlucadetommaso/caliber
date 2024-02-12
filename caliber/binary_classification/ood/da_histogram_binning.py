import numpy as np
from scipy.stats import expon


class DistanceAwareHistogramBinningBinaryClassificationModel:
    def __init__(
        self,
        n_prob_bins: int = 10,
        n_dist_bins: int = 10,
        conf_distance: float = 0.95,
        min_prob_bin: float = 0.0,
    ):
        self.n_prob_bins = n_prob_bins
        self.n_dist_bins = n_dist_bins
        self.conf_distance = conf_distance
        self._min_prob_bin = min_prob_bin
        self._params = None
        self._prob_bin_edges = None
        self._dist_bin_edges = None
        self._cdf = None
        self._max_distance = None
        self._quantile_distance = None

    def fit(self, probs: np.ndarray, distances: np.ndarray, targets: np.ndarray):
        self._quantile_distance = np.quantile(distances, self.conf_distance)
        self._max_distance = np.max(distances)
        self._cdf = expon(0, self._max_distance).cdf

        self._prob_bin_edges = self._get_prob_bin_edges()
        self._dist_bin_edges = self._get_dist_bin_edges()

        prob_bin_indices = np.digitize(probs, self._prob_bin_edges)
        dist_bin_indices = np.digitize(distances, self._dist_bin_edges)

        self._params = np.empty((self.n_prob_bins + 1, self.n_dist_bins + 1))

        for i in range(1, self.n_prob_bins + 2):
            for j in range(1, self.n_dist_bins + 2):
                mask = self._get_mask(i, j, prob_bin_indices, dist_bin_indices)
                self._fit_bin(i, j, mask, targets)

    def predict_proba(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        if self._prob_bin_edges is None:
            raise ValueError("Run `fit` first.")

        prob_bin_indices = np.digitize(probs, self._prob_bin_edges)
        dist_bin_indices = np.digitize(distances, self._dist_bin_edges)

        probs = np.copy(probs)

        for i in range(1, self.n_prob_bins + 2):
            for j in range(1, self.n_dist_bins + 2):
                mask = self._get_mask(i, j, prob_bin_indices, dist_bin_indices)
                if not np.isnan(self._params[i - 1, j - 1]):
                    probs[mask] = self._params[i - 1, j - 1]
                else:
                    cdf = self._get_cdf(distances[mask])
                    probs[mask] = (1 - cdf) * probs[mask] + 0.5 * cdf

        return probs

    def predict(self, probs: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs, distances) >= 0.5).astype(int)

    def _get_prob_bin_edges(self) -> np.ndarray:
        return np.linspace(0, 1, self.n_prob_bins + 1)

    def _get_dist_bin_edges(self) -> np.ndarray:
        return np.linspace(0, self._max_distance, self.n_dist_bins + 1)

    def _fit_bin(self, i: int, j: int, mask: np.ndarray, targets: np.ndarray):
        prob_bin = np.mean(mask)
        if prob_bin >= self._min_prob_bin:
            cdf = self._get_cdf(self._dist_bin_edges[j - 1])
            self._params[i - 1, j - 1] = (1 - cdf) * np.mean(targets[mask]) + 0.5 * cdf
        else:
            self._params[i - 1, j - 1] = np.nan

    @staticmethod
    def _get_mask(
        prob_bin_idx: int,
        dist_bin_idx: int,
        prob_bin_indices: np.ndarray,
        dist_bin_indices: np.ndarray,
    ) -> np.ndarray:
        return (prob_bin_indices == prob_bin_idx) * (dist_bin_indices == dist_bin_idx)

    def _get_cdf(self, d: float) -> np.ndarray:
        d -= self._quantile_distance
        return self._cdf(d) * (d > 0)
