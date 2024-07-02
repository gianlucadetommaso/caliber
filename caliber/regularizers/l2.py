import numpy as np

from caliber.regularizers.base import Regularizer


class L2Regularizer(Regularizer):
    def __init__(self, loc: np.ndarray = 0.0):
        self._loc = loc

    def __call__(self, params: np.ndarray) -> float:
        return np.linalg.norm(params - self._loc)
