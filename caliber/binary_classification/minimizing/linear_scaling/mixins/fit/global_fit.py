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
                return [(-3, 3), (0, 4), (0, 4)]
            return [(-3, 3), (0, 4)]
        if self._has_bivariate_slope:
            return [(0, 4), (0, 4)]
        return [(0, 4)]
