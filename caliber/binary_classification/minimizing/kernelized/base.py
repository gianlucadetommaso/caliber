# from typing import Optional

# import numpy as np

# from caliber.binary_classification.metrics.asce import grouped_average_smooth_squared_calibration_error
# from functools import partial
# from scipy.special import softmax
# from scipy.optimize import minimize
# from scipy.stats import norm
# from caliber.binary_classification.base import AbstractBinaryClassificationModel
# from caliber.binary_classification.checks_mixin import BinaryClassificationChecksMixin


# class OneShotKernelizedBinaryClassificationModel(BinaryClassificationChecksMixin, AbstractBinaryClassificationModel):
#     def __init__(
#         self,
#         minimize_options: Optional[dict] = None,
#         n_bins: int = 10,
#         sigma: float = 0.1
#     ):
#         self._n_bins = n_bins
#         self._sigma = sigma
#         self._mesh = np.linspace(0, 1, n_bins + 1)
#         self._params = None
#         self._minimize_options = minimize_options if minimize_options is not None else {}

#     def fit(self, probs: np.ndarray, targets: np.ndarray, groups: Optional[np.ndarray] = None) -> dict:
#         self._check_targets(targets)
#         self._check_probs(probs)

#         if groups is None:
#             groups = np.ones((len(probs), 1))

#         if "x0" not in self._minimize_options:
#             self._minimize_options["x0"] = np.ones((self._n_bins + 1) * groups.shape[1])


#         def _loss_fn(params):
#             return self._loss_fn(targets, self._predict_proba(params, probs, groups), groups)

#         status = minimize(_loss_fn, **self._minimize_options)
#         self._params = status.x
#         return status

#     def predict_proba(self, probs: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
#         self._check_probs(probs)
#         return np.clip(self._predict_proba(self._params, probs, groups), 0, 1)

#     def predict(self, probs: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
#         probs = self._predict_proba(self._params, probs, groups)
#         return np.array(probs > 0.5, dtype=int)

#     def _predict_proba(self, params: np.ndarray, probs: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
#         kernels = [norm.pdf(probs, loc=p, scale=self._sigma) for p in self._mesh]
#         if groups is None:
#             groups = np.ones((len(probs), 1))
#         features = np.array([kernel * groups[:, i] for kernel in kernels for i in range(groups.shape[1])]).T
#         return np.dot(features, softmax(params))

from functools import partial

#     def _loss_fn(self, targets: np.ndarray, probs: np.ndarray, groups: np.ndarray) -> float:
#         return np.dot(np.array(grouped_average_smooth_squared_calibration_error(targets, probs, groups)), np.mean(groups, 0))
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import norm

from caliber.binary_classification.base import AbstractBinaryClassificationModel
from caliber.binary_classification.checks_mixin import BinaryClassificationChecksMixin
from caliber.binary_classification.metrics.asce import (
    average_smooth_squared_calibration_error,
)


class OneShotKernelizedBinaryClassificationModel(
    BinaryClassificationChecksMixin, AbstractBinaryClassificationModel
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        n_bins: int = 10,
        sigma: float = 0.1,
    ):
        self._n_bins = n_bins
        self._loss_fn = partial(average_smooth_squared_calibration_error, sigma=sigma)
        self._sigma = sigma
        self._mesh = np.linspace(0, 1, n_bins + 1)
        self._params = None
        self._minimize_options = (
            minimize_options if minimize_options is not None else {}
        )

    def fit(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> dict:
        self._check_targets(targets)
        self._check_probs(probs)

        def _loss_fn(params):
            return self._loss_fn(targets, self._predict_proba(params, probs, groups))

        if "x0" not in self._minimize_options:
            if groups is None:
                self._minimize_options["x0"] = np.ones(self._n_bins + 1)
            else:
                self._minimize_options["x0"] = np.ones(
                    (self._n_bins + 1) * groups.shape[1]
                )

        status = minimize(_loss_fn, **self._minimize_options)
        self._params = status.x
        return status

    def predict_proba(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self._check_probs(probs)
        return np.clip(self._predict_proba(self._params, probs, groups), 0, 1)

    def predict(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        probs = self._predict_proba(self._params, probs, groups)
        return np.array(probs > 0.5, dtype=int)

    def _predict_proba(
        self, params: np.ndarray, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        kernels = [norm.pdf(probs, loc=p, scale=self._sigma) for p in self._mesh]
        if groups is not None:
            features = np.array(
                [
                    kernel * groups[:, i]
                    for kernel in kernels
                    for i in range(groups.shape[1])
                ]
            ).T
        else:
            features = np.array(kernels).T
        return np.dot(features, softmax(params))
