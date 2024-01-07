import numpy as np

from caliber.binary_classification.base_smooth_fit_mixin import (
    BinaryClassificationSmoothFitMixin,
)


class BinaryClassificationLinearScalingSmoothFitMixin(
    BinaryClassificationSmoothFitMixin
):
    @staticmethod
    def _get_x0() -> np.ndarray:
        return np.array([0.0, 1.0])
