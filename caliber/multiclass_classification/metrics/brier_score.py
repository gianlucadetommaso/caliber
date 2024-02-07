import numpy as np


def brier_score(targets: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return np.sum((np.eye(probs.shape[1])[targets] - probs) ** 2) / len(targets)
