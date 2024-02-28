import numpy as np


class JointECDF:
    def __init__(self, x: np.ndarray):
        self.x = x

    def evaluate(self, z: np.ndarray) -> np.ndarray:
        return np.mean(np.prod(self.x[None] <= z[:, None], axis=2), axis=1)
