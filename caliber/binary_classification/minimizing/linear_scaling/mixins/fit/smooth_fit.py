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
                x0 = np.array([0.0, 1.0, 1.0])
            else:
                x0 = np.array([0.0, 1.0])
        elif self._has_bivariate_slope:
            x0 = np.array([1.0, 1.0])
        else:
            x0 = np.array(1.0)

        if self._num_features > 0:
            x0 = np.concatenate((x0, np.zeros(self._num_features)))

        return x0

    def _get_bounds(self) -> List:
        if self._has_intercept:
            if self._has_bivariate_slope:
                bounds = [(None, None), (0.0, None), (0.0, None)]
            else:
                bounds = [(None, None), (0.0, None)]
        elif self._has_bivariate_slope:
            bounds = [(0.0, None), (0.0, None)]
        else:
            bounds = [(0.0, None)]

        if self._num_features > 0:
            bounds += [(None, None) for _ in range(self._num_features)]

        return bounds
