import numpy as np
from sklearn.metrics import accuracy_score


def expected_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10, min_prob_bin: float = 0.0
) -> float:
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confs, bin_edges)

    ece = 0
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        prob_bin = np.mean(mask)
        if prob_bin > min_prob_bin:
            _preds = preds[mask]
            _confs = confs[mask]
            _targets = targets[mask]
            acc = accuracy_score(_targets, _preds)
            avg_conf = np.mean(_confs)
            ece += prob_bin * np.abs(acc - avg_conf)
    return ece
