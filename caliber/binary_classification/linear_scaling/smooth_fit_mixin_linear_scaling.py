from typing import List, Optional, Tuple

import numpy as np

from caliber.binary_classification.base_smooth_fit_mixin import (
    BinaryClassificationSmoothFitMixin,
)


class BinaryClassificationLinearScalingSmoothFitMixin(
    BinaryClassificationSmoothFitMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "bounds" not in self._minimize_options:
            self._minimize_options["bounds"] = self._get_bounds()

    def _get_x0(self) -> np.ndarray:
        return np.array([0.0, 1.0]) if self._has_intercept else np.array(1.0)

    def _get_bounds(self) -> List:
        return [(None, None), (0.0, None)] if self._has_intercept else [(0.0, None)]
