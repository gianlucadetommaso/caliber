from copy import deepcopy
from typing import Callable, Optional, Tuple, TypedDict, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from caliber.regression.minimizing.base import MinimizingRegressionModel


class MinimizeOptions(TypedDict):
    x0: np.ndarray


class HeteroskedasticRegressionModel(MinimizingRegressionModel):
    def __init__(
        self,
        mean_logstd_predict_fn: Callable[
            [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
        ],
        minimize_options: MinimizeOptions,
    ):
        super().__init__(loss_fn=self._loss_fn, minimize_options=minimize_options)
        self._mean_logstd_predict_fn = mean_logstd_predict_fn

    @staticmethod
    def _loss_fn(
        targets: np.ndarray, mean_preds: np.ndarray, logstd_preds: np.ndarray
    ) -> float:
        return np.sum(
            0.5 * ((targets - mean_preds) / np.exp(logstd_preds)) ** 2 + logstd_preds
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> dict:
        self._check_targets(targets)
        self._check_features(features)

        def _loss_fn(params):
            means, logstds = self._mean_logstd_predict_fn(params, features)
            return self._loss_fn(targets, means, logstds)

        status = minimize(_loss_fn, **self._minimize_options)
        self._params = status.x
        return status

    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        minimize_options = deepcopy(minimize_options) or dict()
        return minimize_options

    @staticmethod
    def _check_features(features: np.ndarray) -> None:
        assert features.ndim == 2

    @staticmethod
    def _check_targets(targets: np.ndarray) -> None:
        assert targets.ndim == 1

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._mean_logstd_predict_fn(self._params, features)[0]

    def predict_std(self, features: np.ndarray) -> np.ndarray:
        return np.exp(self._mean_logstd_predict_fn(self._params, features)[1])

    def predict_var(self, features: np.ndarray) -> np.ndarray:
        return self.predict_std(features) ** 2

    def predict_quantile(self, features: np.ndarray, confidence: float) -> np.ndarray:
        means = self.predict(features)
        stds = self.predict_std(features)
        quantile = stats.norm().ppf(confidence)
        return means + quantile * stds
