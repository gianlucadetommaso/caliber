import numpy as np


def both_quantile_check(quantiles: np.ndarray) -> None:
    if quantiles.ndim != 2 or quantiles.shape[1] != 2:
        raise ValueError(
            "`quantiles` must be a two-dimensional array of quantiles with `quantiles.shape[1]` equal 2."
        )


def lower_quantile_check(quantiles: np.ndarray) -> None:
    if quantiles.ndim != 1:
        raise ValueError(
            "`quantiles` must be a one-dimensional array of lower quantiles."
        )


def upper_quantile_check(quantiles: np.ndarray) -> None:
    if quantiles.ndim != 1:
        raise ValueError(
            "`quantiles` must be a one-dimensional array of upper quantiles."
        )
