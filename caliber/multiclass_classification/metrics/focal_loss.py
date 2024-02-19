import numpy as np


def focal_loss(targets: np.ndarray, probs: np.ndarray, gamma: float = 2.0) -> float:
    class_probs = probs[np.arange(len(probs)), targets]
    class_probs = np.clip(class_probs, 1e-6, 1 - 1e-6)
    return -np.sum((1 - class_probs) ** gamma * np.log(class_probs))
