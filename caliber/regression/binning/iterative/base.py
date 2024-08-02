import abc
from functools import partial
from typing import Any, Callable, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics import mean_pinball_loss

from caliber.regression.base import AbstractRegressionModel
from caliber.regression.constant_shift.base import ConstantQuantileShiftRegressionModel
from caliber.utils.iterative_binning.base import IterativeBinningModel


class IterativeBinningRegressionModel(IterativeBinningModel, AbstractRegressionModel):
    def __init__(
        self,
        confidence: float,
        which_quantile: Literal["both", "lower", "upper"] = "both",
        n_bins: int = 100,
        max_rounds: int = 1000,
        min_prob_bin: float = 0.01,
        split: float = 0.8,
        seed: int = 0,
        bin_types: Tuple[str] = ("<=", ">="),
        bin_model: Optional[Any] = None,
    ):
        self.confidence = confidence
        super().__init__(
            n_bins=n_bins,
            max_rounds=max_rounds,
            min_prob_bin=min_prob_bin,
            split=split,
            seed=seed,
            bin_types=bin_types,
            bin_loss_fn=self._get_absolute_score_bias,
            early_stopping_loss_fn=partial(mean_pinball_loss, alpha=confidence),
            bin_model=bin_model
            or ConstantQuantileShiftRegressionModel(
                loss_fn=self._get_absolute_score_bias,
            ),
        )
        self.which_quantile = which_quantile

    def fit(
        self,
        values: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> dict:
        return super().fit(
            values=self._initialize_score_quantiles(values),
            targets=self._get_scores(values, targets),
            groups=groups,
        )

    def predict(
        self, values: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> np.ndarray:
        score_quantiles = self._predict_values(
            self._initialize_score_quantiles(values), groups
        )
        return self._get_inverse_scores(values, score_quantiles)

    def _predict_bin_values_fn(self, model: Any) -> Callable[[np.ndarray], np.ndarray]:
        return model.predict

    @abc.abstractmethod
    def _get_scores(self, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _get_inverse_scores(
        self, values: np.ndarray, score_quantiles: np.ndarray
    ) -> np.ndarray:
        pass

    @staticmethod
    def _initialize_score_quantiles(values: np.ndarray) -> np.ndarray:
        return np.zeros(len(values))

    def _get_absolute_score_bias(self, scores: np.ndarray, values: np.ndarray) -> float:
        return float(np.abs(np.mean(scores <= values) - self.confidence))

    def _get_bin_edges(self, scores: np.ndarray) -> np.ndarray:
        return np.linspace(scores.min(), scores.max(), self.n_bins + 1)
