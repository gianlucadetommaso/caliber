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

    def _check_features(self, features: np.ndarray) -> None:
        if features is None and self._num_features > 0:
            raise ValueError(
                f"`features` is not passed but the number of features is set to {self._num_features}."
            )
        if features is not None:
            if features.ndim != 2:
                raise ValueError("`features` must be a two-dimensional array.")
            if self._num_features != features.shape[1]:
                raise ValueError(
                    f"`features.shape[1]` and {self._num_features} must match, but {features.shape[1]} and {self._num_features} were found."
                )
