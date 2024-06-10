import numpy as np

from caliber.multiclass_classification.minimizing.smooth_fit_mixin import (
    SmoothFitMulticlassClassificationMixin,
)


class LinearScalingSmoothFitMulticlassClassificationMixin(
    SmoothFitMulticlassClassificationMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_x0(self) -> np.ndarray:
        if self._has_intercept:
            if self._has_shared_intercept:
                intercept = np.array([0.0])
            else:
                intercept = np.zeros(self._n_classes)
        if self._has_shared_slope:
            slope = np.array([1.0])
        elif self._has_cross_slopes:
            slope = np.eye(self._n_classes).flatten()
        else:
            slope = np.ones(self._n_classes)

        if self._has_intercept:
            return np.concatenate((intercept, slope))
        return slope
