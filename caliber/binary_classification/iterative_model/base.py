import abc
from copy import deepcopy
from typing import Any, Tuple

import numpy as np


class BinaryClassificationIterativeModel:
    def __init__(self, model: Any, n_rounds: int, seed: int = 0):
        self._model = model
        self.n_rounds = n_rounds
        self._models = []
        self._rng = np.random.default_rng(seed)

    def fit(self, features: np.ndarray, targets: np.ndarray):
        self._models = []
        model = deepcopy(self._model)
        for t in range(self.n_rounds):
            self._fit(model, features, targets)
            self._models.append(model)
            model, features, targets = self._update(model, features, targets)

    @abc.abstractmethod
    def _fit(self, model: Any, features: np.ndarray, targets: np.ndarray):
        pass

    @abc.abstractmethod
    def _predict_proba(self, model: Any, features: np.ndarray):
        pass

    @abc.abstractmethod
    def _update(
        self, model: Any, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def _predict(self, model: Any, features: np.ndarray) -> np.ndarray:
        pass
