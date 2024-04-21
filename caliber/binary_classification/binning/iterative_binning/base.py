from typing import Any, Callable, Optional, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss

from caliber.binary_classification.constant_shift.model_bias.base import (
    ModelBiasConstantShiftBinaryClassificationModel,
)
from caliber.binary_classification.metrics.bias import absolute_model_bias
from caliber.utils.iterative_binning.base import IterativeBinningModel


class IterativeBinningBinaryClassificationModel(IterativeBinningModel):
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
        bin_model: Any = ModelBiasConstantShiftBinaryClassificationModel(step_size=0.1),
    ):
        super().__init__(
            n_bins=n_bins,
            max_rounds=max_rounds,
            min_prob_bin=min_prob_bin,
            split=split,
            seed=seed,
            bin_types=bin_types,
            bin_loss_fn=bin_loss_fn,
            early_stopping_loss_fn=early_stopping_loss_fn,
            bin_model=bin_model,
        )

    def fit(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> dict:
        return super().fit(values=probs, targets=targets, groups=groups)

    def predict_proba(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return super()._predict_values(values=probs, groups=groups)

    def predict(
        self, probs: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return (self.predict_proba(probs, groups) > 0.5).astype(int)

    def _predict_bin_values_fn(self, model: Any) -> Callable[[np.ndarray], np.ndarray]:
        return model.predict_proba

    def _get_bin_edges(self, targets: np.ndarray) -> np.ndarray:
        return np.linspace(0, 1, self.n_bins + 1)
