from typing import List, Tuple

from caliber.binary_classification.minimizing.linear_scaling.mixins.regularizers.l2 import (
    L2RegularizerMixin,
)
from caliber.binary_classification.minimizing.mixins.fit.brute_fit import (
    BruteFitBinaryClassificationMixin,
)


class LinearScalingBruteFitBinaryClassificationMixin(
    L2RegularizerMixin, BruteFitBinaryClassificationMixin
):
    def _get_ranges(self) -> List[Tuple]:
        if self._has_intercept:
            if self._has_bivariate_slope:
                return [(-2, 2), (0, 4), (0, 4)]
            return [(-2, 2), (0, 4)]
        if self._has_bivariate_slope:
            return [(0, 4), (0, 4)]
        return [(0, 4)]

    @staticmethod
    def _get_Ns() -> int:
        return 200
