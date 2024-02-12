import numpy as np


class HistogramBinningMulticlassClassificationModel:
    def __init__(self, n_prob_bins: int = 10, min_prob_bin: float = 0.0):
        self.n_prob_bins = n_prob_bins
        self._min_prob_bin = min_prob_bin
        self._params = None
        self._prob_bin_edges = None
        self._n_classes = None

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        self._n_classes = probs.shape[1]
        self._prob_bin_edges = self._get_prob_bin_edges()
        prob_bin_indices = np.digitize(probs, self._prob_bin_edges)
        top_class_indices = np.argmax(probs, axis=1)

        self._params = np.empty((self.n_prob_bins + 1, self._n_classes))

        for i in range(1, self.n_prob_bins + 2):
            for c in range(self._n_classes):
                mask = self._get_mask(i, c, prob_bin_indices, top_class_indices)
                self._fit_bin(i, c, mask, targets)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if self._prob_bin_edges is None:
            raise ValueError("Run `fit` first.")
        if probs.shape[1] != self._n_classes:
            raise ValueError(
                "The number of classes when fitting and predicting must be the same."
            )

        prob_bin_indices = np.digitize(probs, self._prob_bin_edges)
        top_class_indices = np.argmax(probs, axis=1)

        probs = np.copy(probs)

        for i in range(1, self.n_prob_bins + 2):
            for c in range(0, self._n_classes):
                mask = self._get_mask(i, c, prob_bin_indices, top_class_indices)
                if not np.isnan(self._params[i - 1, c]):
                    probs[mask, c] = self._params[i - 1, c]
        return probs

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(probs), axis=1)

    def _get_prob_bin_edges(self) -> np.ndarray:
        return np.linspace(0, 1, self.n_prob_bins + 1)

    def _fit_bin(self, i: int, c: int, mask: np.ndarray, targets: np.ndarray):
        prob_bin = np.mean(mask)
        if prob_bin >= self._min_prob_bin:
            self._params[i - 1, c] = np.mean(targets[mask] == c)
        else:
            self._params[i - 1, c] = np.nan

    @staticmethod
    def _get_mask(
        prob_bin_idx: int,
        class_idx: int,
        prob_bin_indices: np.ndarray,
        top_class_indices: np.ndarray,
    ) -> np.ndarray:
        return (prob_bin_indices[:, class_idx] == prob_bin_idx) * (
            top_class_indices == class_idx
        )
