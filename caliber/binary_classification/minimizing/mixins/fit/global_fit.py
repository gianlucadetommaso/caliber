import abc
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import shgo


class GlobalFitBinaryClassificationMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._regularizer = self._init_regularizer()

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        self._check_targets(targets)
        self._check_probs(probs)

        def _loss_fn(params: np.ndarray):
            regularizer = self._regularizer(params)
            return (
                self._loss_fn(targets, self._get_output_for_loss(params, probs))
                + self._lam * regularizer
            )

        results = shgo(_loss_fn, **self._minimize_options)
        if results.success:
            self._params = results.x
        else:
            self._params = self._regularizer._loc

    @staticmethod
    def _get_bounds() -> List[Tuple]:
        pass

    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        minimize_options = deepcopy(minimize_options) or dict()
        if "bounds" not in minimize_options:
            minimize_options["bounds"] = self._get_bounds()
        return minimize_options
