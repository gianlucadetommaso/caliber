import numpy as np
from scipy.stats import norm


def average_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10, min_prob_bin: float = 0.0
) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges)

    asce = 0
    for i in range(1, n_bins + 2):
        mask = bin_indices == i
        prob_bin = np.mean(mask)
        if prob_bin > min_prob_bin:
            asce += prob_bin * np.mean(targets[mask] - probs[mask]) ** 2
    return asce


# def average_smooth_squared_calibration_error(
#     targets: np.ndarray, probs: np.ndarray, n_bins: int = 10, sigma: float = 0.1
# ) -> float:
#     bin_edges = np.linspace(0, 1, n_bins + 1)
#     bin_indices = np.digitize(probs, bin_edges)
    
#     assce = 0.
#     for i, p in enumerate(bin_edges):
#         mask = bin_indices == i + 1
#         mean_p = np.mean(mask)
        
#         kernels = norm.pdf(probs, loc=p, scale=sigma)
#         assce += np.mean(kernels * (targets - probs) ** 2)
#     return assce
def average_smooth_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10, sigma: float = 0.1
) -> float:
    assce = 0
    for p in np.linspace(0, 1, n_bins + 1):
        kernels = norm.pdf(probs, loc=p, scale=sigma)
        assce += np.mean(kernels * (targets - probs) ** 2)
    assce /= n_bins + 1
    return assce


def grouped_average_squared_calibration_error(
    targets: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 10,
    min_prob_bin: float = 0.0,
) -> list[float]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges)

    gasce = []
    for j in range(groups.shape[1]):
        group = groups[:, j]
        
        mean_g = np.mean(group)
        if mean_g == 0:
            gasce.append(np.nan)
        else:
            gasce.append(0.0)
            
            for i in range(1, n_bins + 2):
                mask = bin_indices == i
                mean_pg = np.mean(group * mask)
                if mean_pg > min_prob_bin:
                    gasce[-1] += np.mean(group[mask] * (targets[mask] - probs[mask])) ** 2
            gasce[-1] /= mean_g

    return gasce


def grouped_average_smooth_squared_calibration_error(
    targets: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 10,
    min_prob_bin: float = 0.0, 
    sigma: float = 0.1
) -> list[float]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges)

    gasce = []
    for j in range(groups.shape[1]):
        group = groups[:, j]
        
        mean_g = np.mean(group)
        if mean_g == 0:
            gasce.append(np.nan)
        else:
            gasce.append(0.0)
            
            for i, p in enumerate(bin_edges):
                mask = bin_indices == i + 1
                mean_pg = np.mean(mask * group)
                
                if mean_pg > min_prob_bin:
                    kernels = norm.pdf(probs, loc=p, scale=sigma)
                    gasce[-1] += mean_pg * np.mean(kernels * group * (targets - probs)) ** 2
            gasce[-1] /= mean_g

    return gasce