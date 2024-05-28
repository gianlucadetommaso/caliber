import abc
from typing import Optional

import numpy as np

from caliber.regression.base import AbstractRegressionModel


class ConformalizedScoreRegressionModel(AbstractRegressionModel, abc.ABC):
    def __init__(
        self,
        confidence: float,
    ):
        super().__init__()
        self.confidence = confidence

    def fit(self, scores: np.ndarray, targets: Optional[np.ndarray] = None) -> None:
        size = len(scores)
        adjusted_confidence = np.ceil((size + 1) * self.confidence) / size
        self._params = np.quantile(scores, adjusted_confidence)

    def threshold(self) -> np.ndarray:
        return self._params
