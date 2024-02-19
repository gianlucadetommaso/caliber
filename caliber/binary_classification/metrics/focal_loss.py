import numpy as np


def focal_loss(targets: np.ndarray, probs: np.ndarray, gamma: float = 2.0) -> float:
    new_probs = np.clip(probs, 1e-6, 1 - 1e-6)
    new_probs = np.where(targets == 1, new_probs, 1 - new_probs)
    return -np.sum((1 - new_probs) ** gamma * np.log(new_probs))
