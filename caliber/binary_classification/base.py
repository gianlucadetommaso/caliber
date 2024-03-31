import abc

import numpy as np


class AbstractBinaryClassificationModel:
    def __init__(self):
        self._params = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        pass
