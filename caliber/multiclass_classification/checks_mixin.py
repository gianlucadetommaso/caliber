import numpy as np


class MulticlassClassificationChecksMixin:
    @staticmethod
    def _check_targets(targets: np.ndarray) -> None:
        if targets.dtype not in ["int32", "int64"]:
            raise ValueError("`targets` must be an array of integers.")
        if targets.ndim != 1:
            raise ValueError("`targets` must be a one-dimensional array.")

    @staticmethod
    def _check_probs(probs: np.ndarray) -> None:
        if probs.ndim != 2:
            raise ValueError("`probs` must be a two-dimensional array.")
        if np.sum(probs > 1) or np.sum(probs < 0):
            raise ValueError("All values in `probs` must be between 0 and 1.")
