from typing import List

import numpy as np

from caliber.binary_classification.minimizing.mixins.fit.smooth_fit import (
    SmoothFitBinaryClassificationMixin,
)


class LinearScalingSmoothFitBinaryClassificationMixin(
    SmoothFitBinaryClassificationMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "bounds" not in self._minimize_options:
            self._minimize_options["bounds"] = self._get_bounds()

    def _get_x0(self) -> np.ndarray:
        if self._has_intercept:
            if self._has_bivariate_slope:
                return np.array([0.0, 1.0, 1.0])
            return np.array([0.0, 1.0])
        if self._has_bivariate_slope:
            return np.array([1.0, 1.0])
        return np.array(1.0)

    def _get_bounds(self) -> List:
        if self._has_intercept:
            if self._has_bivariate_slope:
                return [(None, None), (0.0, None), (0.0, None)]
            return [(None, None), (0.0, None)]
        if self._has_bivariate_slope:
            return [(0.0, None), (0.0, None)]
        return [(0.0, None)]
