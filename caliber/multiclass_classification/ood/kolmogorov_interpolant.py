from functools import partial
from typing import Any, Callable, List, Optional

import numpy as np
from scipy import stats
from scipy.special import kolmogorov


class KolmogorovInterpolantMulticlassClassificationModel:
    def __init__(
        self,
        model: Optional[Any] = None,
        reduct_fn: Callable[[np.ndarray], np.ndarray] = partial(
            np.mean, axis=1, keepdims=True
        ),
    ):
        self.model = model
        self.reduct_fn = reduct_fn
        self._train_mv_ecdf = None

    def fit(self, probs: np.ndarray, embeddings: np.ndarray, targets: np.ndarray):
        if self.model is not None:
            self.model.fit(probs, targets)
        self._train_mv_ecdf = self._get_ecdf(embeddings)
        self._train_size = len(embeddings)

    def predict_proba(self, probs: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        probs = np.copy(probs)
        if self.model is not None:
            probs = self.model.predict_proba(probs)

        mv_ecdf = self._get_ecdf(embeddings)
        d = np.abs(
            self._eval_ecdf(mv_ecdf, embeddings)
            - self._eval_ecdf(self._train_mv_ecdf, embeddings)
        )

        w = self.reduct_fn(kolmogorov(np.sqrt(len(embeddings)) * d))
        return w * probs + (1 - w) / probs.shape[1]

    def predict(self, probs: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(probs, embeddings), axis=1)

    @staticmethod
    def _get_ecdf(embeddings) -> List[Callable]:
        return [
            stats.ecdf(embeddings[:, i]).cdf.evaluate
            for i in range(embeddings.shape[1])
        ]

    @staticmethod
    def _eval_ecdf(mv_ecdf: List[Callable], embeddings) -> float:
        return np.array([ecdf(embeddings[:, i]) for i, ecdf in enumerate(mv_ecdf)]).T


# class KolmogorovInterpolantMulticlassClassificationModel:
#     def __init__(self, model: Optional[Any] = None, reduct_fn: Callable[[np.ndarray], np.ndarray] = partial(np.mean, axis=1, keepdims=True)):
#         self.model = model
#         self.reduct_fn = reduct_fn
#         self._train_mv_ecdf = None
#         self._train_size = None
#
#     def fit(self, probs: np.ndarray, embeddings: np.ndarray, targets: np.ndarray):
#         if self.model is not None:
#             self.model.fit(probs, targets)
#         self._train_mv_ecdf = self._get_ecdf(embeddings)
#         self._train_size = len(embeddings)
#
#     def predict_proba(self, probs: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
#         probs = np.copy(probs)
#         if self.model is not None:
#             probs = self.model.predict_proba(probs)
#
#         vals = self._eval_ecdf(self._train_mv_ecdf, embeddings)
#         t = np.max(np.maximum(vals, 1 - vals), 1, keepdims=True) / (self._train_size + 1)
#         # t = np.maximum(vals, 1 - vals)# / (self._train_size + 1)
#         n_classes = probs.shape[1]
#
#         # w = self.reduct_fn(kolmogorov(d))
#         # w = 1 - self.reduct_fn((1 - kolmogorov(d)) ** d.shape[1])
#         # w = np.minimum((self._train_size + 1) * n_classes * np.exp(-2 * self._train_size * t ** 2), 1)
#         # w = 1 - (1 - np.exp(-2 * self._train_size * t ** 2)) ** (self._train_size * n_classes)
#         # r = 0.01
#         # w = self.reduct_fn(np.minimum(1, kolmogorov(t) / r))
#         w = 1 - (1 - kolmogorov(np.sqrt(self._train_size) * t))# ** n_classes
#         return w * probs + (1 - w) / n_classes
#
#     def predict(self, probs: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
#         return np.argmax(self.predict_proba(probs, embeddings), axis=1)
#
#     @staticmethod
#     def _get_ecdf(embeddings) -> List[Callable]:
#         return [
#             stats.ecdf(embeddings[:, i]).cdf.evaluate
#             for i in range(embeddings.shape[1])
#         ]
#
#     @staticmethod
#     def _eval_ecdf(mv_ecdf: List[Callable], embeddings) -> float:
#         return np.array([ecdf(embeddings[:, i]) for i, ecdf in enumerate(mv_ecdf)]).T
