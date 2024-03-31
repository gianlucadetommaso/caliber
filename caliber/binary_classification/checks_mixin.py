import numpy as np


class BinaryClassificationChecksMixin:
    @staticmethod
    def _check_targets(targets: np.ndarray) -> None:
        if set(targets) not in [{0, 1}, {0}, {1}]:
            raise ValueError("`targets` must must include only values in {0, 1}.")
        if targets.ndim != 1:
            raise ValueError("`targets` must be a one-dimensional array.")

    @staticmethod
    def _check_probs(probs: np.ndarray) -> None:
        if probs.ndim != 1:
            raise ValueError("`probs` must be a one-dimensional array.")
        if np.sum(probs > 1) or np.sum(probs < 0):
            raise ValueError("All values in `probs` must be between 0 and 1.")
