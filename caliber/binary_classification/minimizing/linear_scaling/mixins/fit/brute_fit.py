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
                bounds = [(-2, 2), (0, 4), (0, 4)]
            else:
                bounds = [(-2, 2), (0, 4)]
        elif self._has_bivariate_slope:
            bounds = [(0, 4), (0, 4)]
        else:
            bounds = [(0, 4)]

        if self._num_features > 0:
            bounds += [(None, None) for _ in range(self._num_features)]

        return bounds

    @staticmethod
    def _get_Ns() -> int:
        return 200
