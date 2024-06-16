import numpy as np

from caliber.binary_classification.utils.knee_point import knee_point


def knee_point_distance(
    targets: np.ndarray, probs: np.ndarray, n_thresholds: int = 100
) -> float:
    precision, recall, _ = knee_point(probs, targets, n_thresholds)
    return (1 - precision) ** 2 + (1 - recall) ** 2
