from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from caliber.regression.conformal_regression.base import AbstractRegressionModel
from caliber.utils.functional import maybe_squeeze


class CVPlusRegressionModel(AbstractRegressionModel):
    """
    A conformalised bootstrap model based on cross-validation (CV).
    Given inputs and targets, it trains an arbitrary model multiple times with CV and stores the prediction error on the left-out inputs.
    At prediction time, it provides a confidence interval around the mean predicted, with quantiles corrected via the errors stored at training time.

    Given a coverage level `alpha`, a target variable `Y` and the predicted confidence interval `[Q1, Q2]`,
    if the training and test sets are IID the algorithm ensures that `P(Y in [Q1, Q2]) >= alpha`.
    """

    def __init__(
        self,
        model: Any,
        coverage: float,
        num_folds: int = 5,
        seed: int = 0,
        cv_prediction: bool = False,
        max_validation_fold_size: int = 1000,
    ) -> None:
        """

        Args:
            model (Any): An instantiated model class with `fit` and `predict` methods.
            coverage (float): A coverage value between 0 and 1.
                For example, `coverage=0.95` means that the target variable will be expected to lay within the confidence interval 95% of the times.
            num_folds (int, optional): The number of CV folds. Defaults to 5.
            seed (int, optional): The random seed. Defaults to 0.
            cv_prediction (bool, optional): Whether to predict the mean prediction using the cv models or rather a single model. Defaults to False.
        """
        self._model = model
        self._coverage = coverage
        self._num_folds = num_folds
        self._cv_prediction = cv_prediction
        self._rng = np.random.default_rng(seed)
        self._max_validation_fold_size = max_validation_fold_size

    def fit(
        self, inputs: NDArray[np.float64], targets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        num_inputs = len(inputs)
        fold_size = num_inputs // self._num_folds
        perm = self._rng.choice(num_inputs, size=num_inputs, replace=False)
        fold_indices = [
            [perm[j * fold_size + i] for i in range(fold_size)]
            for j in range(self._num_folds)
        ]
        fold_indices[-1].extend(perm[self._num_folds * fold_size :].tolist())

        self._models = []
        self._cv_errors = []
        for i in tqdm(range(self._num_folds), desc="Cross-Validation"):
            train_indices = sum(fold_indices[:i] + fold_indices[i + 1 :], [])
            val_indices = fold_indices[i][: self._max_validation_fold_size]
            train_inputs, train_targets = (
                inputs[train_indices, :],
                targets[train_indices],
            )
            val_inputs, val_targets = inputs[val_indices], targets[val_indices]

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

            self._cv_errors.append(np.abs(val_targets - val_preds))

            self._models.append(model)

        if not self._cv_prediction:
            self._model.fit(inputs, targets)

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._cv_prediction:
            preds = np.zeros(len(inputs))
            for model in self._models:
                preds += model.predict(inputs)
            preds /= self._loo_size
            return preds

        return self._model.predict(inputs)

    def predict_quantiles(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        lefts_list = []
        rights_list = []
        for i, model in enumerate(self._models):
            preds_i = model.predict(inputs)
            if preds_i.ndim > 2:
                raise ValueError(
                    "Predictions are expected to be one or two dimensional arrays."
                )
            if preds_i.ndim == 1:
                preds_i = preds_i[:, None]

            lefts_list.append(preds_i[None] - self._cv_errors[i][:, None])
            rights_list.append(preds_i[None] + self._cv_errors[i][:, None])

        lefts = np.concatenate(lefts_list, axis=0)
        rights = np.concatenate(rights_list, axis=0)

        qleft = np.quantile(lefts, q=1 - self._coverage, axis=0)
        qright = np.quantile(rights, q=self._coverage, axis=0)
        return np.concatenate((qleft, qright), axis=1)
