import abc
from typing import Any

import numpy as np


class AbstractMulticlassClassificationModel:
    def __init__(
        self,
    ):
        self._params = None
        self._n_classes = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        pass

    @abc.abstractmethod
    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        pass
