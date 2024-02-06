import logging
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


class IterativeFittingBinaryClassificationModel:
    def __init__(
        self,
        max_rounds: int = 1000,
        split: float = 0.8,
        seed: int = 0,
        loss_fn: Callable[[np.ndarray, np.ndarray], float] = log_loss,
        early_stopping_loss_fn: Callable[[np.ndarray, np.ndarray], float] = log_loss,
        n_bins: int = 10,
    ):
        self.max_rounds = max_rounds
        self.split = split
        self.loss_fn = loss_fn
        self.early_stopping_loss_fn = early_stopping_loss_fn
        self._rng = np.random.default_rng(seed)
        self.n_bins = n_bins
        self._params = None
        self._bin_edges = None

    def fit(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> dict:
        n_data = len(probs)
        perm = self._rng.choice(n_data, n_data, replace=False)
        calib_size = int(np.ceil(n_data * self.split))
        calib_probs, val_probs = probs[perm[:calib_size]], probs[perm[calib_size:]]
        calib_targets, val_targets = (
            targets[perm[:calib_size]],
            targets[perm[calib_size:]],
        )
        calib_groups, val_groups = (
            groups[perm[:calib_size]],
            groups[perm[calib_size:]],
        )

        self._bin_edges = self._get_bin_edges()
        calib_features = self._get_features(calib_probs, calib_groups)
        n_features = calib_features.shape[1]

        self._params = []
        val_losses = [self.early_stopping_loss_fn(val_targets, val_probs)]

        def loss_fn(_params: np.ndarray) -> float:
            return self.loss_fn(
                calib_targets, self._predict_proba(_params, calib_probs, calib_features)
            )

        for t in range(self.max_rounds):
            params = minimize(loss_fn, np.zeros(n_features)).x

            val_features = self._get_features(val_probs, val_groups)
            val_probs = self._predict_proba(params, val_probs, val_features)
            val_losses.append(self.early_stopping_loss_fn(val_targets, val_probs))

            if val_losses[-1] >= val_losses[-2]:
                logging.info(
                    f"Early stopping triggered after {t} rounds. The loss started increasing on the validation data."
                )
                break

            calib_probs = self._predict_proba(params, calib_probs, calib_features)
            calib_features = self._get_features(calib_probs, calib_groups)
            self._params.append(params)

        return dict(n_iter=len(val_losses) - 1, val_losses=val_losses)

    def predict_proba(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._params is None:
            raise ValueError("Run `fit` first.")
        probs = np.copy(probs)
        for params in self._params:
            features = self._get_features(probs, groups)
            probs = self._predict_proba(params, probs, features)
        return probs

    def predict(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return (self.predict_proba(probs, groups) >= 0.5).astype(int)

    @staticmethod
    def _predict_proba(
        params: np.ndarray, probs: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        return np.clip(probs + np.dot(features, params), 0, 1)

    def _get_features(
        self,
        probs: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ):
        bin_indices = np.digitize(probs, self._bin_edges)
        features = np.stack(
            [bin_indices == i for i in range(1, self.n_bins + 2)], axis=1
        )
        if groups is not None:
            features = np.concatenate([features * g[:, None] for g in groups.T], axis=1)
        return features

    def _get_bin_edges(self):
        return np.linspace(0, 1, self.n_bins + 1)
