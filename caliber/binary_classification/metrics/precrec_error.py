import numpy as np


def precision_recall_error(
    targets: np.ndarray, probs: np.ndarray, threshold: float
) -> float:
    preds = probs >= threshold
    mean_targets = np.mean(targets)
    mean_preds = np.mean(preds)

    recall = np.mean(targets * preds) / mean_targets
    lower_bound = max(
        threshold,
        1 - threshold,
        threshold * mean_preds / mean_targets,
        1 - threshold * (1 - mean_preds) / mean_targets,
    )
    return max(0, lower_bound - recall)
