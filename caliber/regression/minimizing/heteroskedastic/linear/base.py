from typing import Tuple

import numpy as np

from caliber.regression.minimizing.heteroskedastic.base import (
    HeteroskedasticRegressionModel,
    MinimizeOptions,
)


class HeteroskedasticLinearRegressionModel(HeteroskedasticRegressionModel):
    def __init__(
        self,
        minimize_options: MinimizeOptions,
    ):
        super().__init__(
            mean_logstd_predict_fn=self._linear_mean_logstd_predict_fn,
            minimize_options=minimize_options,
        )

    def _linear_mean_logstd_predict_fn(
        self, params: np.ndarray, inputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean_params, logstd_params = np.split(params, 2)
        mean_preds = mean_params[0] + np.dot(inputs, mean_params[1:])
        logstd_preds = logstd_params[0] + np.dot(inputs, logstd_params[1:])
        return mean_preds, logstd_preds
