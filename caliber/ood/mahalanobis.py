from typing import Optional

import numpy as np
from scipy import stats
from scipy.linalg.lapack import dtrtri

from caliber.binary_classification.base import AbstractBinaryClassificationModel


class MahalanobisBinaryClassificationModel(AbstractBinaryClassificationModel):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self._mean, self._chol = None, None

    def fit(self, embeddings: np.ndarray, targets: Optional[np.ndarray] = None):
        if targets is None:
            self._mean, self._chol = self._get_mean_and_chol(embeddings)
        else:
            self._mean, self._chol = dict(), dict()
            unique_targets = np.unique(targets)
            for y in unique_targets:
                _embeddings = embeddings[targets == y]
                self._mean[y], self._chol[y] = self._get_mean_and_chol(_embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        if isinstance(self._mean, dict):
            dists = []
            for mean in self._mean.values():
                dists.append(np.linalg.norm(embeddings - mean[None], axis=1))
            dists = np.stack(dists, axis=1)
            indices = np.argmin(dists, 1)
            modes = np.array(list(self._mean.keys()))[indices]
            unique_modes = np.unique(modes)

            probs = np.zeros(len(embeddings))
            for y in unique_modes:
                indices = np.where(modes == y)[0]
                _embeddings = embeddings[indices]
                probs[indices] = self._get_probs(
                    _embeddings, self._mean[y], self._chol[y]
                )
            return probs

        return self._get_probs(embeddings, self._mean, self._chol)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return (self.predict_proba(embeddings) > self.threshold).astype(int)

    @staticmethod
    def _get_mean_and_chol(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(embeddings, 0)
        cov = np.cov(embeddings.T)
        chol = np.linalg.cholesky(cov)
        chol = dtrtri(chol, lower=True)[0].T
        return mean, chol

    def _get_probs(
        self, embeddings: np.ndarray, mean: np.ndarray, chol: np.ndarray
    ) -> np.ndarray:
        transformed_embeddings = self._get_transformed_embeddings(
            embeddings, mean, chol
        )
        return stats.chi2(embeddings.shape[1]).cdf(transformed_embeddings)

    @staticmethod
    def _get_transformed_embeddings(
        embeddings: np.ndarray, mean: np.ndarray, chol: np.ndarray
    ) -> np.ndarray:
        return np.sum(np.matmul(embeddings - mean[None], chol) ** 2, axis=1)
