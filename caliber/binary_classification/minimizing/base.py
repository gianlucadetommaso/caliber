import abc
from typing import Callable, Optional

import numpy as np

from caliber.binary_classification.base import AbstractBinaryClassificationModel
from caliber.binary_classification.minimizing.mixins.pred_from_probs_mixin import (
    PredFromProbsBinaryClassificationMixin,
)


class MinimizingBinaryClassificationModel(
    PredFromProbsBinaryClassificationMixin, AbstractBinaryClassificationModel
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        threshold: float,
        minimize_options: Optional[dict] = None,
        num_features: int = 0,
    ):
        self._num_features = num_features
        super().__init__(threshold=threshold)
        self._loss_fn = loss_fn
        self._params = None
        self._minimize_options = self._config_minimize_options(minimize_options)

    def predict_proba(
        self, probs: np.ndarray, features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self._check_probs(probs)
        self._check_features(features)
        return self._predict_proba(self._params, probs, features)

    @abc.abstractmethod
    def _get_output_for_loss(
        self,
        params: np.ndarray,
        probs: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        pass

    @staticmethod
    @abc.abstractmethod
    def _predict_proba(
        params: np.ndarray, probs: np.ndarray, features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pass
