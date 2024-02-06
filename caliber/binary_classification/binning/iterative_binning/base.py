import logging
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss

from caliber.binary_classification.constant_shift.model_bias.base import (
    ModelBiasBinaryClassificationConstantShift,
)
from caliber.binary_classification.metrics.bias import absolute_model_bias


class IterativeBinningBinaryClassificationModel:
    def __init__(
        self,
        n_bins: int = 100,
        max_rounds: int = 1000,
        min_prob_bin: float = 0.01,
        split: float = 0.8,
        seed: int = 0,
        bin_types: Tuple[str] = ("<=", ">="),
        bin_loss_fn: Callable[[np.ndarray, np.ndarray], float] = absolute_model_bias,
        early_stopping_loss_fn: Callable[
            [np.ndarray, np.ndarray], float
        ] = brier_score_loss,
        bin_model: Any = ModelBiasBinaryClassificationConstantShift(step_size=0.1),
    ):
        self.n_bins = n_bins
        self.max_rounds = max_rounds
        self.min_prob_bin = min_prob_bin
        self.split = split
        self._rng = np.random.default_rng(seed)
        self.bin_loss_fn = bin_loss_fn
        self.early_stopping_loss_fn = early_stopping_loss_fn
        self.bin_model = bin_model
        self._bin_types = bin_types
        self._supported_bin_types = ["=", "<=", ">="]
        self._params = None
        self._bin_edges = None

    def fit(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> dict:
        n_data = len(probs)
        if groups is None:
            groups = self._initialize_groups(n_data)

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
        self._params = []
        val_losses = [self.early_stopping_loss_fn(val_targets, val_probs)]

        for t in range(self.max_rounds):
            bin_idx, bin_type, group_idx, calib_mask = self._get_worst_bin(
                calib_targets, calib_probs, calib_groups
            )

            calib_prob_bin = np.mean(calib_mask)
            if calib_prob_bin < self.min_prob_bin:
                logging.info(
                    f"Early stopping triggered after {t} rounds. "
                    f"The bin with largest loss got smaller than min_prob_bin={self.min_prob_bin}."
                )
                break

            model = self._fit_bin_model(calib_probs, calib_targets, calib_mask)
            val_mask = self._get_mask(
                val_probs, bin_idx, bin_type, val_groups[:, group_idx]
            )
            val_probs[val_mask] = self._predict_bin_proba(model, val_probs, val_mask)

            val_losses.append(self.early_stopping_loss_fn(val_targets, val_probs))

            if val_losses[-1] >= val_losses[-2]:
                logging.info(
                    f"Early stopping triggered after {t} rounds. The loss started increasing on the validation data."
                )
                break

            calib_probs[calib_mask] = self._predict_bin_proba(
                model, calib_probs, calib_mask
            )
            self._update_params(bin_idx, bin_type, group_idx, model)

        return dict(n_iter=len(val_losses) - 1, val_losses=val_losses)

    def predict_proba(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._params is None:
            raise ValueError("Run `fit` first.")
        if groups is None:
            groups = self._initialize_groups(len(probs))

        probs = np.copy(probs)
        for bin_idx, bin_type, group_idx, model in self._params:
            mask = self._get_mask(probs, bin_idx, bin_type, groups[:, group_idx])
            probs[mask] = self._predict_bin_proba(model, probs, mask)
        return probs

    def predict(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return (self.predict_proba(probs, groups) >= 0.5).astype(int)

    def _weighted_bin_loss_fn(
        self, targets: np.ndarray, probs: np.ndarray, mask: np.ndarray
    ) -> float:
        prob_mask = np.mean(mask)
        if prob_mask == 0.0 or prob_mask < self.min_prob_bin:
            return 0.0
        return prob_mask * self.bin_loss_fn(targets[mask], probs[mask])

    def _get_worst_bin(
        self, targets: np.ndarray, probs: np.ndarray, groups: np.ndarray
    ) -> Tuple[int, str, int, np.ndarray]:
        bin_indices = np.digitize(probs, self._bin_edges)
        n_groups = groups.shape[1]

        def _fun(i: int, _bin_type: str, j: int):
            mask = self._get_mask(probs, i, _bin_type, groups[:, j], bin_indices)
            return self._weighted_bin_loss_fn(targets, probs, mask)

        group_idx, idx_bin_type, idx_bin_idx = np.unravel_index(
            np.argmax(
                [
                    [
                        [_fun(i, bt, j) for i in range(1, self.n_bins + 2)]
                        for bt in self._bin_types
                    ]
                    for j in range(n_groups)
                ]
            ),
            (n_groups, len(self._bin_types), self.n_bins + 1),
        )
        bin_idx = idx_bin_idx + 1
        bin_type = self._bin_types[idx_bin_type]
        return (
            bin_idx,
            bin_type,
            group_idx,
            self._get_mask(probs, bin_idx, bin_type, groups[:, group_idx], bin_indices),
        )

    def _fit_bin_model(
        self, probs: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> Any:
        model = deepcopy(self.bin_model)
        model.fit(probs[mask], targets[mask])
        return model

    def _get_mask(
        self,
        probs: np.ndarray,
        bin_idx: int,
        bin_type: str,
        group: np.ndarray,
        bin_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if bin_indices is None:
            bin_indices = np.digitize(probs, self._bin_edges)
        if bin_type == "=":
            return (bin_indices == bin_idx) * group
        elif bin_type == ">=":
            return (bin_indices >= bin_idx) * group
        elif bin_type == "<=":
            return (bin_indices <= bin_idx) * group
        else:
            raise ValueError(f"bin_type={bin_type} not recognized.")

    @staticmethod
    def _predict_bin_proba(
        model: Any, probs: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        return model.predict_proba(probs[mask])

    def _update_params(
        self, bin_idx: int, bin_type: str, group_idx: int, model: Any
    ) -> None:
        self._params.append((bin_idx, bin_type, group_idx, model))

    def _get_bin_edges(self):
        return np.linspace(0, 1, self.n_bins + 1)

    @staticmethod
    def _initialize_groups(size: int):
        return np.ones((size, 1)).astype(bool)
