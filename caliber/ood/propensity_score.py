import numpy as np
from numpy.typing import NDArray
from xgboost import XGBClassifier

from caliber.binary_classification.base import AbstractBinaryClassificationModel
from caliber.binary_classification.minimizing.linear_scaling.calibration.beta import (
    BetaBinaryClassificationModel,
)


class PropensityScoreEstimationModel:
    def __init__(
        self,
        binary_classification_estimation_model: AbstractBinaryClassificationModel = XGBClassifier(
            objective="binary:logistic"
        ),
        binary_classification_calibration_model: AbstractBinaryClassificationModel
        | None = BetaBinaryClassificationModel(),
        calib_frac: float = 0.5,
        clip_range: tuple[float, float] = (0.05, 0.95),
        seed: int = 0,
    ) -> None:
        """
        A propensity score estimation model. Given inputs from a source and a target distribution,
        and targets describing which distributions they belong to,
        the model fits and then calibrates the probability that the input belongs to the distribution with target label equal 1.
        Finally, it predicts the propensity score as an odds ratio, clipped to make sure it does not explode.

        Args:
            binary_classification_estimation_model (_type_, optional): A binary classification model to classify inputs to source and
                target distributions. The model needs to include a `fit` and a `predict_proba` methods, with analogous signature as in
                standard Scikit-Learn binary classification models. Defaults to XGBClassifier(objective='binary:logistic').
            binary_classification_calibration_model (AbstractBinaryClassificationModel | None, optional): A binary classification model
                to calibrate the probability returned by the estimation model. Defaults to BetaBinaryClassificationModel().
            calib_frac (float, optional): The fraction of the data reserved for calibration. Defaults to 0.5.
            clip_range (tuple[float, float], optional): The range to clip the propensity score within. Defaults to (0.05, 0.95).
            seed (int, optional): a random seed. Defaults to 0.
        """
        self._binary_classification_estimation_model = (
            binary_classification_estimation_model
        )
        self._binary_classification_calibration_model = (
            binary_classification_calibration_model
        )
        self._calib_frac = calib_frac
        self._clip_range = clip_range
        self._rng = np.random.default_rng(seed)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> None:
        """
        Fits the propensity score model.

        Args:
            X (NDArray[np.float64]): Inputs from source and target distributions.
            y (NDArray[np.int64]): Binary targets describing which distribution an input belongs to.
        """
        if self._binary_classification_calibration_model is not None:
            size = len(X)
            train_size = int(size * (1 - self._calib_frac))
            perm = self._rng.choice(size, size, replace=False)
            train_perm, calib_perm = perm[:train_size], perm[train_size:]
            X_train, X_calib = X[train_perm], X[calib_perm]
            y_train, y_calib = y[train_perm], y[calib_perm]

            self._binary_classification_estimation_model.fit(X_train, y_train)
            p_calib = self._binary_classification_estimation_model.predict_proba(
                X_calib
            )
            p_calib = self._reshape_proba(p_calib)

            self._binary_classification_calibration_model.fit(p_calib, y_calib)
        else:
            self._binary_classification_estimation_model.fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predicts the propensity score.

        Args:
            X (NDArray[np.float64]): Test inputs.

        Returns:
            NDArray[np.float64]: the estimated propensity score for each input.
        """
        p = self._binary_classification_estimation_model.predict_proba(X)
        p = self._reshape_proba(p)
        if self._binary_classification_calibration_model is not None:
            p = self._binary_classification_calibration_model.predict_proba(p)
        p = np.clip(p, *self._clip_range)
        return p / (1 - p)

    def _reshape_proba(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        if p.ndim > 1:
            if p.shape[1] == 2:
                p = p[:, 1]
            elif p.shape[1] == 1:
                p = p.squeeze(1)
            else:
                raise ValueError(
                    "The binary classification estimation model does not seem to return binary class probabilities."
                )
        return p
