from typing import List, Tuple

from caliber.binary_classification.minimizing.base_brute_fit_mixin import (
    BruteFitBinaryClassificationMixin,
)


class LinearScalingBruteFitBinaryClassificationMixin(BruteFitBinaryClassificationMixin):
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
