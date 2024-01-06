import abc

import numpy as np
from scipy.special import expit, logit

from caliber.binary_classification.base import CustomBinaryClassificationModel


class CustomBinaryClassificationLinearScaling(CustomBinaryClassificationModel, abc.ABC):
    @staticmethod
    def _predict_proba(params: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return expit(params[0] + params[1] * logit(probs))
