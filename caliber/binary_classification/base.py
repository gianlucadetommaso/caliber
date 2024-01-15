import abc
from typing import Callable, Optional

import numpy as np


class CustomBinaryClassificationModel:
    def __init__(
        self,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        minimize_options: Optional[dict] = None,
    ):
        self._loss_fn = loss_fn
        self._params = None
        self._minimize_options = self._config_minimize_options(minimize_options)

    @abc.abstractmethod
    def fit(self, probs: np.ndarray, targets: np.ndarray):
        pass

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return self._predict_proba(self._params, probs)

    @staticmethod
    @abc.abstractmethod
    def _predict_proba(params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def predict(self, probs: np.ndarray):
        self._check_probs(probs)
        pass

    @abc.abstractmethod
    def _get_output_for_loss(self, params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _check_targets(targets: np.ndarray) -> None:
        if set(targets) not in [{0, 1}, {0}, {1}]:
            raise ValueError("`targets` must must include only values in {0, 1}.")
        if targets.ndim != 1:
            raise ValueError("`targets` must be a one-dimensional array.")

    @staticmethod
    def _check_probs(probs: np.ndarray) -> None:
        if probs.ndim != 1:
            raise ValueError("`probs` must be a one-dimensional array.")
        if np.sum(probs > 1) or np.sum(probs < 0):
            raise ValueError("All values in `probs` must be between 0 and 1.")

    @abc.abstractmethod
    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        pass
