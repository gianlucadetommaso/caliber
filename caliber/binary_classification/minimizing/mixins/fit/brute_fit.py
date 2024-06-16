import abc
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brute, fmin


class BruteFitBinaryClassificationMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._regularizer = self._init_regularizer()

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        self._check_targets(targets)
        self._check_probs(probs)

        def _loss_fn(params):
            regularizer = self._regularizer(params)
            return (
                self._loss_fn(targets, self._get_output_for_loss(params, probs))
                + self._lam * regularizer
            )

        self._params = brute(_loss_fn, **self._minimize_options)

    @staticmethod
    def _get_ranges() -> List[Tuple]:
        pass

    @staticmethod
    def _get_Ns() -> int:
        pass

    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        minimize_options = deepcopy(minimize_options) or dict()
        if "ranges" not in minimize_options:
            minimize_options["ranges"] = self._get_ranges()
        if "Ns" not in minimize_options:
            minimize_options["Ns"] = self._get_Ns()
        if "finish" not in minimize_options:
            minimize_options["finish"] = fmin
        return minimize_options
