import numpy as np
from sklearn.calibration import calibration_curve


def average_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    binned_ey, binned_probs = calibration_curve(
        targets, probs, n_bins=n_bins, strategy="quantile"
    )
    return float(np.mean((binned_ey - binned_probs) ** 2))
