import numpy as np


def expected_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10, min_prob_bin: float = 0.0
) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges)

    ece = 0
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        prob_bin = np.mean(mask)
        if prob_bin > min_prob_bin:
            _probs = probs[mask]
            _targets = targets[mask]
            acc = np.mean((_probs >= 0.5) == _targets)
            conf = np.mean(np.maximum(1 - _probs, _probs))
            ece += prob_bin * np.abs(acc - conf)
    return ece
