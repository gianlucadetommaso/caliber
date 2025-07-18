import numpy as np
from numpy.typing import NDArray


def both_quantile_check(quantiles: NDArray[np.float64], y_dim: int) -> None:
    if quantiles.ndim == 2 and quantiles.shape[1] != 2 * y_dim:
        raise ValueError("`quantiles.shape[1]` must be the same as `2 * y_dim`.")


def single_quantile_check(quantiles: NDArray[np.float64], y_dim: int) -> None:
    if quantiles.ndim == 1 and y_dim != 1:
        raise ValueError("If `quantiles.ndim==1`, then `y_dim` must be 1.")
    if quantiles.ndim == 2 and quantiles.shape[1] != y_dim:
        raise ValueError("`quantiles.shape[1]` must be the same as `y_dim`.")
