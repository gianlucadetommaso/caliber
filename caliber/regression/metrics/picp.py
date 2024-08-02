from typing import Literal

import numpy as np


def prediction_interval_coverage_probability(
    targets: np.ndarray,
    bounds: np.ndarray,
    which_quantile: Literal["both", "lower", "upper"],
) -> np.ndarray:
    if targets.ndim > 1:
        raise ValueError("`targets` must be a 1-dimensional array.")
    if which_quantile == "both":
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                "`bounds` must be a two-dimensional array of bounds with `bounds.shape[1]` equal 2."
            )
        conds = (targets <= bounds[:, 1]) * (targets >= bounds[:, 0])
    elif which_quantile == "lower":
        if bounds.ndim != 1:
            raise ValueError(
                "`bounds` must be a one-dimensional array of right quantiles."
            )
        conds = targets <= bounds
    elif which_quantile == "upper":
        if bounds.ndim != 1:
            raise ValueError(
                "`bounds` must be a one-dimensional array of left quantiles."
            )
        conds = targets >= bounds
    else:
        raise ValueError(f"`which_quantile={which_quantile}` not recognized.")
    return np.mean(conds)
