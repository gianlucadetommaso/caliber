import numpy as np


def pinball_loss(
    targets: np.ndarray, quantiles: np.ndarray, confidence: float
) -> float:
    diffs = targets - quantiles
    conds = diffs > 0
    return np.mean(
        confidence * diffs * conds + (1 - confidence) * (1 - diffs) * (1 - conds)
    )
