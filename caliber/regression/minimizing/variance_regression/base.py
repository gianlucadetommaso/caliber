from copy import deepcopy
from typing import Callable, Optional, Tuple, TypedDict, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from caliber.regression.minimizing.base import MinimizingRegressionModel


class MinimizeOptions(TypedDict):
    x0: np.ndarray


class LogStdRegressionModel(MinimizingRegressionModel):
    def __init__(
        self,
        mean_model,
        logstd_predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        minimize_options: MinimizeOptions,
    ):
        super().__init__(loss_fn=self._loss_fn, minimize_options=minimize_options)
        self._mean_model = mean_model
        self._logstd_predict_fn = logstd_predict_fn

    @staticmethod
    def _loss_fn(
        targets: np.ndarray, mean_preds: np.ndarray, logstd_preds: np.ndarray
    ) -> float:
        return np.sum(
            0.5 * ((targets - mean_preds) / np.exp(logstd_preds)) ** 2 + logstd_preds
        )

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> dict:
        self._check_targets(targets)
        self._check_inputs(inputs)
        means = self._mean_model.predict(inputs)

        def _loss_fn(params):
            logstds = self._logstd_predict_fn(params, inputs)
            return self._loss_fn(targets, means, logstds)

        status = minimize(_loss_fn, **self._minimize_options)
        self._params = status.x
        return status

    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        minimize_options = deepcopy(minimize_options) or dict()
        return minimize_options

    @staticmethod
    def _check_inputs(inputs: np.ndarray) -> None:
        assert inputs.ndim == 2

    @staticmethod
    def _check_targets(targets: np.ndarray) -> None:
        assert targets.ndim == 1

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self._mean_model.predict(inputs)

    def predict_std(self, inputs: np.ndarray) -> np.ndarray:
        return np.exp(self._logstd_predict_fn(self._params, inputs))

    def predict_var(self, inputs: np.ndarray) -> np.ndarray:
        return self.predict_std(inputs) ** 2

    def predict_quantile(self, inputs: np.ndarray, confidence: float) -> np.ndarray:
        means = self.predict(inputs)
        stds = self.predict_std(inputs)
        quantile = stats.norm().ppf(confidence)
        return means + quantile * stds
