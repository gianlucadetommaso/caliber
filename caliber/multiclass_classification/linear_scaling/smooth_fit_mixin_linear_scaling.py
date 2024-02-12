import numpy as np

from caliber.multiclass_classification.base_smooth_fit_mixin import (
    MulticlassClassificationSmoothFitMixin,
)


class MulticlassClassificationLinearScalingSmoothFitMixin(
    MulticlassClassificationSmoothFitMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_x0(self) -> np.ndarray:
        return np.ones(1) if self._has_shared_slope else np.ones(self._n_classes)
