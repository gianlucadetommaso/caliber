from typing import Literal

import numpy as np


def prediction_interval_coverage_probability(
    targets: np.ndarray,
    quantiles: np.ndarray,
    which_quantile: Literal["both", "lower", "upper"],
) -> np.ndarray:
    if targets.ndim > 1:
        raise ValueError("`targets` must be a 1-dimensional array.")
    if which_quantile == "both":
        if quantiles.ndim != 2 or quantiles.shape[1] != 2:
            raise ValueError(
                "`quantiles` must be a two-dimensional array of quantiles with `quantiles.shape[1]` equal 2."
            )
        conds = (targets <= quantiles[:, 1]) * (targets >= quantiles[:, 0])
    elif which_quantile == "lower":
        if quantiles.ndim != 1:
            raise ValueError(
                "`quantiles` must be a one-dimensional array of lower quantiles."
            )
        conds = targets >= quantiles
    elif which_quantile == "upper":
        if quantiles.ndim != 1:
            raise ValueError(
                "`quantiles` must be a one-dimensional array of upper quantiles."
            )
        conds = targets <= quantiles
    else:
        raise ValueError(f"`which_quantile={which_quantile}` not recognized.")
    return np.mean(conds)
