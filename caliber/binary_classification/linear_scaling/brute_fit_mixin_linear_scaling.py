from typing import List, Tuple

from caliber.binary_classification.base_brute_fit_mixin import (
    BinaryClassificationBruteFitMixin,
)


class BinaryClassificationLinearScalingBruteFitMixin(BinaryClassificationBruteFitMixin):
    @staticmethod
    def _get_ranges() -> List[Tuple]:
        return [(-2, 2), (-2, 2)]

    @staticmethod
    def _get_Ns() -> int:
        return 200
