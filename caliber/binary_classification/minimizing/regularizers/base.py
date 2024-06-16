import abc

import numpy as np


class Regularizer:
    @abc.abstractmethod
    def __call__(self, params: np.ndarray) -> float:
        pass
