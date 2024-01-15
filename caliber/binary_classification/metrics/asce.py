import numpy as np
from scipy.stats import norm
from sklearn.calibration import calibration_curve


def average_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    binned_ey, binned_probs = calibration_curve(
        targets, probs, n_bins=n_bins, strategy="quantile"
    )
    return float(np.mean((binned_ey - binned_probs) ** 2))


def average_smooth_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10, smoothness: float = 0.1
) -> float:
    assce = 0
    for i in range(1, n_bins + 1):
        kernels = norm.pdf(probs, i, smoothness)
        assce += np.mean(kernels) * np.mean(kernels * (targets - probs)) ** 2
    return assce
