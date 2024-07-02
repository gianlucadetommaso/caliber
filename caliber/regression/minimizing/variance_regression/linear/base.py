from typing import Tuple

import numpy as np

from caliber.regression.minimizing.variance_regression.base import (
    LogStdRegressionModel,
    MinimizeOptions,
)


class LogStdLinearRegressionModel(LogStdRegressionModel):
    def __init__(
        self,
        mean_model,
        minimize_options: MinimizeOptions,
    ):
        super().__init__(
            mean_model=mean_model,
            logstd_predict_fn=self._linear_logstd_predict_fn,
            minimize_options=minimize_options,
        )

    @staticmethod
    def _linear_logstd_predict_fn(params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        logstd_preds = params[0] + np.dot(inputs, params[1:])
        return logstd_preds
