import numpy as np


def prediction_interval_average_length(
    bounds: np.ndarray,
) -> np.ndarray:
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(
            "`bounds` must be a two-dimensional array of bounds with `bounds.shape[1]` equal 2."
        )
    return np.mean(bounds[:, 1] - bounds[:, 0])
