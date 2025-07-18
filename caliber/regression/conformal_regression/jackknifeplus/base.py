from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from caliber.regression.conformal_regression.base import AbstractRegressionModel
from caliber.utils.functional import maybe_squeeze


class JackknifePlusRegressionModel(AbstractRegressionModel):
    """
    A conformalised bootstrap model based on leave-one-out (LOO).
    Given inputs and targets, it trains an arbitrary model multiple times with LOO and stores the prediction error on the left-out input.
    At prediction time, it provides a confidence interval around the mean predicted, with quantiles corrected via the errors stored at training time.

    Given a coverage level `alpha`, a target variable `Y` and the predicted confidence interval `[Q1, Q2]`,
    if the training and test sets are IID the algorithm ensures that `P(Y in [Q1, Q2]) >= alpha`.
    """

    def __init__(
        self,
        model: Any,
        coverage: float,
        loo_size: int = 100,
        seed: int = 0,
        loo_prediction: bool = False,
    ) -> None:
        """

        Args:
            model (Any): An instantiated model class with `fit` and `predict` methods.
            coverage (float): A coverage value between 0 and 1.
                For example, `coverage=0.95` means that the target variable will be expected to lay within the confidence interval 95% of the times.
            loo_size (int, optional): The number of leave-one-out (LOO) validations. Defaults to 100.
            seed (int, optional): The random seed. Defaults to 0.
            loo_prediction (bool, optional): Whether to predict the mean prediction using the leave-one-out models or rather a single model. Defaults to False.
        """
        self._model = model
        self._coverage = coverage
        self._loo_size = loo_size
        self._loo_prediction = loo_prediction
        self._rng = np.random.default_rng(seed)

    def fit(
        self, inputs: NDArray[np.float64], targets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        indices = self._rng.choice(len(inputs), size=self._loo_size, replace=False)

        self._models = []
        self._loo_errors: NDArray[np.float64] = np.zeros(
            (self._loo_size, targets.shape[1] if targets.ndim == 2 else 1)
        )
        for i, val_idx in enumerate(tqdm(indices, desc="Leave-One-Out")):
            train_indices = [idx for idx in range(len(inputs)) if idx != val_idx]
            train_inputs, train_targets = inputs[train_indices], targets[train_indices]
            val_inputs, val_targets = inputs[None, val_idx], targets[None, val_idx]

            model = deepcopy(self._model)
            model.fit(train_inputs, train_targets)

            val_preds = model.predict(val_inputs)
            if val_preds.ndim > 2:
                raise ValueError(
                    "Predictions are expected to be one or two dimensional arrays."
                )
            if val_preds.ndim == 1:
                val_preds = val_preds[:, None]
            if val_targets.ndim == 1:
                val_targets = val_targets[:, None]

            self._loo_errors[i] = np.abs(val_targets - val_preds)[0]

            self._models.append(model)

        if not self._loo_prediction:
            self._model.fit(inputs, targets)

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._loo_prediction:
            preds = np.zeros(len(inputs))
            for model in self._models:
                preds += model.predict(inputs)
            preds /= self._loo_size
            return preds

        return self._model.predict(inputs)

    def predict_quantiles(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        loo_preds = np.zeros((self._loo_size, len(inputs), self._loo_errors.shape[1]))
        for i, model in enumerate(self._models):
            preds = model.predict(inputs)
            if preds.ndim > 2:
                raise ValueError(
                    "Predictions are expected to be one or two dimensional arrays."
                )
            if preds.ndim == 1:
                preds = preds[:, None]
            loo_preds[i] = preds

        left = loo_preds - self._loo_errors[:, None]
        right = loo_preds + self._loo_errors[:, None]

        qleft = np.quantile(left, q=1 - self._coverage, axis=0)
        qright = np.quantile(right, q=self._coverage, axis=0)
        return np.concatenate((qleft, qright), axis=1)
