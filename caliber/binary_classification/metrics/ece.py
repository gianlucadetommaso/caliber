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
            _probs = probs[mask]
            _targets = targets[mask]
            acc = np.mean((_probs >= 0.5) == _targets)
            conf = np.mean(np.maximum(1 - _probs, _probs))
            ece += prob_bin * np.abs(acc - conf)
    return ece
