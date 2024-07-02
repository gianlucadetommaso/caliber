import numpy as np

from caliber.regularizers.l2 import L2Regularizer


class L2RegularizerMixin:
    def _init_regularizer(self) -> L2Regularizer:
        if self._has_intercept:
            if self._has_bivariate_slope:
                loc = np.array([0.0, 1.0, 1.0])
            else:
                loc = np.array([0.0, 1.0])
        elif self._has_bivariate_slope:
            loc = np.array([1.0, 1.0])
        else:
            loc = np.array(1.0)
        return L2Regularizer(loc)
