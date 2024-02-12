import numpy as np


def false_positive_rate(targets: np.ndarray, preds: np.ndarray) -> float:
    return np.mean(preds * (1 - targets)) / (1 - np.mean(targets))


def false_negative_rate(targets: np.ndarray, preds: np.ndarray) -> float:
    return np.mean((1 - preds) * targets) / np.mean(targets)


def true_positive_rate(targets: np.ndarray, preds: np.ndarray) -> float:
    return 1 - false_negative_rate(targets, preds)


def true_negative_rate(targets: np.ndarray, preds: np.ndarray) -> float:
    return 1 - false_positive_rate(targets, preds)
