import abc
from typing import Callable, Optional

import numpy as np

from caliber.multiclass_classification.base import AbstractMulticlassClassificationModel
from caliber.multiclass_classification.pred_from_probs_mixin import (
    PredFromProbsMulticlassClassificationMixin,
)


class MinimizingMulticlassClassificationModel(
    PredFromProbsMulticlassClassificationMixin, AbstractMulticlassClassificationModel
):
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        minimize_options: Optional[dict] = None,
    ):
        super().__init__()
        self._loss_fn = loss_fn
        self._minimize_options = self._config_minimize_options(minimize_options)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return self._predict_proba(self._params, probs)

    @staticmethod
    @abc.abstractmethod
    def _predict_proba(params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _get_output_for_loss(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        pass
