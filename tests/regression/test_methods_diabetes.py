import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, QuantileRegressor

from caliber import (
    ConformalizedQuantileRegressionModel,
    CVPlusRegressionModel,
    IterativeBinningMeanRegressionModel,
    IterativeBinningQuantileRegressionModel,
    JackknifePlusRegressionModel,
)
from data import load_diabetes_data

CONFIDENCE = 0.95
TRAIN_VAL_SPLIT = 0.5
QUANTILE_MODEL_CLS = QuantileRegressor
PRED_MODEL_CLS = LinearRegression

train_inputs, test_inputs, train_targets, test_targets = load_diabetes_data()

train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

confidences = [0.5 * (1 - CONFIDENCE), 0.5 * (1 + CONFIDENCE)]
val_quantiles, test_quantiles = [], []
for confidence in confidences:
    model = QUANTILE_MODEL_CLS(quantile=confidence)
    model.fit(train_inputs, train_targets)
    val_quantiles.append(model.predict(val_inputs))
    test_quantiles.append(model.predict(test_inputs))
val_quantiles = np.stack(val_quantiles, axis=1)
test_quantiles = np.stack(test_quantiles, axis=1)

pred_model = PRED_MODEL_CLS()
pred_model.fit(train_inputs, train_targets)
val_preds = pred_model.predict(val_inputs)
test_preds = pred_model.predict(test_inputs)

ONE_D_METHODS = {
    "cqr": ConformalizedQuantileRegressionModel(
        confidence=CONFIDENCE,
    ),
    "ibqr": IterativeBinningQuantileRegressionModel(
        confidence=CONFIDENCE,
    ),
    "ibmr": IterativeBinningMeanRegressionModel(
        confidence=CONFIDENCE,
    ),
    "jkp": JackknifePlusRegressionModel(model=pred_model, coverage=0.95, loo_size=3),
    "cvp": CVPlusRegressionModel(model=pred_model, coverage=0.95, num_folds=3),
}

MULTI_D_METHODS = {
    "cqr": ConformalizedQuantileRegressionModel(
        confidence=CONFIDENCE,
    ),
    "jkp": JackknifePlusRegressionModel(model=pred_model, coverage=0.95, loo_size=3),
    "cvp": CVPlusRegressionModel(model=pred_model, coverage=0.95, num_folds=3),
}


@pytest.mark.parametrize("m_name", ONE_D_METHODS)
def test_1d_method(m_name):
    m = ONE_D_METHODS[m_name]
    if m_name not in ["ibmr", "jkp", "cvp"]:
        m.fit(val_quantiles, val_targets)
        calib_test_quantiles = m.predict(test_quantiles)
    elif m_name == "ibmr":
        m.fit(val_preds, val_targets)
        calib_test_quantiles = m.predict(test_preds)
    elif m_name in ["jkp", "cvp"]:
        m.fit(val_inputs, val_targets)
        calib_test_quantiles = m.predict_quantiles(test_inputs)
    check_1d_quantiles(calib_test_quantiles)


NUM_DIMS = 3
multi_d_val_quantiles = np.tile(val_quantiles, (1, NUM_DIMS))
multi_d_test_quantiles = np.tile(test_quantiles, (1, NUM_DIMS))
multi_d_val_targets = np.broadcast_to(
    val_targets[:, None], (len(val_targets), NUM_DIMS)
)
multi_d_test_targets = np.broadcast_to(
    test_targets[:, None], (len(test_targets), NUM_DIMS)
)


@pytest.mark.parametrize("m_name", MULTI_D_METHODS)
def test_multi_d_method(m_name):
    m = MULTI_D_METHODS[m_name]
    if m_name in ["jkp", "cvp"]:
        m.fit(val_inputs, multi_d_val_targets)
        multi_d_calib_test_quantiles = m.predict_quantiles(test_inputs)
    else:
        m.fit(multi_d_val_quantiles, multi_d_val_targets)
        multi_d_calib_test_quantiles = m.predict(multi_d_test_quantiles)
    check_multi_d_quantiles(multi_d_calib_test_quantiles, multi_d_test_targets)


def check_1d_quantiles(quantiles: NDArray[np.float64]) -> None:
    assert quantiles.ndim == 2 and quantiles.shape[1] == 2


def check_multi_d_quantiles(
    quantiles: NDArray[np.float64], y: NDArray[np.float64]
) -> None:
    assert quantiles.ndim == 2
    assert quantiles.shape[1] == 2 * y.shape[1]
    assert quantiles.shape[0] == y.shape[0]
