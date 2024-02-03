import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


class GroupConditionalUnbiasedBinaryClassificationModel:
    def __init__(
        self,
    ):
        self._params = None

    def fit(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        groups: np.ndarray,
    ) -> dict:
        self._params = LogisticRegression()
        status = self._params.fit(self._get_features(probs, groups), targets)

        return status

    def predict_proba(self, probs: np.ndarray, groups: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise ValueError("Run `fit` first.")
        return self._params.predict_proba(self._get_features(probs, groups))[:, 1]

    def predict(self, probs: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs, groups) >= 0.5).astype(int)

    @staticmethod
    def _predict_proba(
        params: np.ndarray, probs: np.ndarray, groups: np.ndarray
    ) -> np.ndarray:
        return np.clip(probs + np.dot(groups, params), 0.0, 1.0)

    @staticmethod
    def _get_features(probs: np.ndarray, groups: np.ndarray):
        return np.concatenate((logit(probs)[:, None], groups), axis=1)
