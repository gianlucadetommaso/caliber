import numpy as np

from caliber.binary_classification.base_smooth_fit_mixin import (
    BinaryClassificationSmoothFitMixin,
)


class BinaryClassificationLinearScalingSmoothFitMixin(
    BinaryClassificationSmoothFitMixin
):
    def _get_x0(self) -> np.ndarray:
        return np.array([0.0, 1.0]) if self._has_intercept else np.array(1.0)
