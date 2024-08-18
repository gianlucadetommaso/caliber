from typing import List

import numpy as np

from caliber.binary_classification.minimizing.mixins.fit.smooth_fit import (
    SmoothFitBinaryClassificationMixin,
)


class KernelisedFitBinaryClassificationMixin(SmoothFitBinaryClassificationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_x0(self) -> np.ndarray:
        return np.ones(self._n_bins + 1)
