import numpy as np
from numpy.typing import NDArray


def maybe_squeeze(a: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    if a.shape[axis] == 1:
        return a.squeeze(axis)
    return a
