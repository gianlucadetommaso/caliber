from typing import Callable

import numpy as np
from scipy.stats import norm


def average_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges)

    def _fun(i: int) -> float:
        mask = bin_indices == i + 1
        prob_bin = np.mean(mask)
        return (
            prob_bin * np.mean(targets[mask] - probs[mask]) ** 2
            if prob_bin > 0.0
            else 0.0
        )

    return trapezoidal_rule(_fun, np.arange(1, n_bins + 2))


def average_smooth_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, n_bins: int = 10, sigma: float = 0.1
) -> float:
    def _fun(p: float) -> float:
        kernels = norm.pdf(probs, loc=p, scale=sigma)
        return np.mean(kernels * (targets - probs)) ** 2 / np.mean(kernels)

    return trapezoidal_rule(_fun, np.linspace(0, 1, n_bins + 1))


def grouped_average_squared_calibration_error(
    targets: np.ndarray, probs: np.ndarray, groups: np.ndarray, n_bins: int = 10
) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges)

    def _fun(i: int) -> float:
        mask = bin_indices == i + 1
        mean_gp = np.mean(group * mask)
        return (
            np.mean(group * mask * (targets - probs)) ** 2 / mean_gp
            if mean_gp > 0.0
            else 0.0
        )

    gasce = []
    for j in range(groups.shape[1]):
        group = groups[:, j]
        mean_g = np.mean(group)

        if mean_g > 0:
            gasce.append(trapezoidal_rule(_fun, np.arange(0, n_bins + 1)))
            gasce[-1] /= mean_g
        else:
            gasce.append(np.nan)
    return gasce


def grouped_average_smooth_squared_calibration_error(
    targets: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 10,
    sigma: float = 0.1,
) -> list[float]:
    def _fun(p: float) -> float:
        kernels = norm.pdf(probs, loc=p, scale=sigma)
        mean_gk = np.mean(group * kernels)
        return np.mean(group * kernels * (targets - probs)) ** 2 / mean_gk

    gassce = []
    for j in range(groups.shape[1]):
        group = groups[:, j]
        mean_g = np.mean(group)

        if mean_g > 0:
            gassce.append(trapezoidal_rule(_fun, np.linspace(0, 1, n_bins + 1)))
            gassce[-1] / mean_g
        else:
            gassce.append(np.nan)
    return gassce


def trapezoidal_rule(fun: Callable, args: np.array) -> float:
    size = len(args)
    area = 0
    for i, arg in enumerate(args):
        delta = fun(arg)
        if i == 0 or i == size - 1:
            delta /= 2
        area += delta
    area *= 1 / (size - 1)
    return area
