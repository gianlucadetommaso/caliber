import abc
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize


class BinaryClassificationSmoothFitMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> dict:
        self._check_targets(targets)
        self._check_probs(probs)

        def _loss_fn(params):
            return self._loss_fn(targets, self._get_output_for_loss(params, probs))

        status = minimize(_loss_fn, **self._minimize_options)
        self._params = status.x
        return status

    @staticmethod
    def _get_x0() -> Union[np.ndarray, float]:
        pass

    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        minimize_options = deepcopy(minimize_options) or dict()
        if "x0" not in minimize_options:
            minimize_options["x0"] = self._get_x0()
        return minimize_options
