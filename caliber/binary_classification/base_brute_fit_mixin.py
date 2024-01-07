import abc
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brute


class BinaryClassificationBruteFitMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, probs: np.ndarray, targets: np.ndarray):
        self._check_targets(targets)
        self._check_probs(probs)

        def _loss_fn(params):
            return self._loss_fn(targets, self._get_output_for_loss(params, probs))

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
        return minimize_options
