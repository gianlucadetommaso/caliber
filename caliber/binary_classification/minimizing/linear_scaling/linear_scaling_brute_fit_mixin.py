from typing import List, Tuple

from caliber.binary_classification.minimizing.base_brute_fit_mixin import (
    BruteFitBinaryClassificationMixin,
)


class LinearScalingBruteFitBinaryClassificationMixin(BruteFitBinaryClassificationMixin):
    def _get_ranges(self) -> List[Tuple]:
        return [(-2, 2), (0, 4)] if self._has_intercept else [(0, 4)]

    @staticmethod
    def _get_Ns() -> int:
        return 200
