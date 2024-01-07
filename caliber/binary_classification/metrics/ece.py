import numpy as np


def expected_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    bin_edges = np.linspace(0, 1, n_bins)
    bin_indices = np.digitize(probs, bin_edges)

    ece = 0
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        prob_bin = np.mean(mask)
        if prob_bin > 0:
            acc = np.mean((probs[mask] >= 0.5) == targets[mask])
            conf = np.mean(probs[mask])
            ece += prob_bin * np.abs(acc - conf)
    return ece
