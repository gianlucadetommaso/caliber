from typing import List, Tuple

from caliber.binary_classification.minimizing.linear_scaling.mixins.regularizers.l2 import (
    L2RegularizerMixin,
)
from caliber.binary_classification.minimizing.mixins.fit.global_fit import (
    GlobalFitBinaryClassificationMixin,
)


class LinearScalingGlobalFitBinaryClassificationMixin(
    L2RegularizerMixin, GlobalFitBinaryClassificationMixin
):
    def _get_bounds(self) -> List[Tuple]:
        if self._has_intercept:
            if self._has_bivariate_slope:
                bounds = [(-3, 3), (0, 4), (0, 4)]
            else:
                bounds = [(-3, 3), (0, 4)]
        elif self._has_bivariate_slope:
            bounds = [(0, 4), (0, 4)]
        else:
            bounds = [(0, 4)]

        if self._num_features > 0:
            bounds += [(None, None) for _ in range(self._num_features)]

        return bounds
