import numpy as np
from sklearn import metrics


def log_loss(targets: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return metrics.log_loss(targets, probs, labels=np.arange(probs.shape[1]))
