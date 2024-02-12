import numpy as np


def average_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10, min_prob_bin: float = 0.0
) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    top_class_indices = np.argmax(probs, axis=1)
    n_classes = probs.shape[1]
    one_hot_targets = np.eye(n_classes)[targets]

    asce = 0
    for c in range(n_classes):
        bin_indices = np.digitize(probs[:, c], bin_edges)
        for i in range(1, n_bins + 1):
            mask = (bin_indices == i) * (top_class_indices == c)
            prob_bin = np.mean(mask)
            if prob_bin > min_prob_bin:
                class_probs = probs[mask, c]
                class_targets = one_hot_targets[mask, c]
                asce += prob_bin * np.mean(class_targets - class_probs) ** 2
    return asce
