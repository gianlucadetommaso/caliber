import abc

import numpy as np

from caliber.multiclass_classification.checks_mixin import (
    MulticlassClassificationChecksMixin,
)


class PredFromProbsMulticlassClassificationMixin(MulticlassClassificationChecksMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        pass

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(probs), 1)
