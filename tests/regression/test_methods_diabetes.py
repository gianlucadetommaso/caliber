import numpy as np
import pytest
from sklearn.linear_model import QuantileRegressor

from caliber import ConformalizedQuantileRegressionModel
from data import load_diabetes_data

CONFIDENCE = 0.95
TRAIN_VAL_SPLIT = 0.5
MODEL_CLS = QuantileRegressor

train_inputs, test_inputs, train_targets, test_targets = load_diabetes_data()

train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

confidences = [1 - CONFIDENCE, CONFIDENCE]
val_quantiles, test_quantiles = [], []
for confidence in confidences:
    model = MODEL_CLS(quantile=confidence)
    model.fit(train_inputs, train_targets)
    val_quantiles.append(model.predict(val_inputs))
    test_quantiles.append(model.predict(test_inputs))
val_quantiles = np.stack(val_quantiles, axis=1)
test_quantiles = np.stack(test_quantiles, axis=1)

METHODS = {
    "cqr": ConformalizedQuantileRegressionModel(
        confidence=CONFIDENCE,
    ),
}


@pytest.mark.parametrize("m", list(METHODS.values()))
def test_method(m):
    m.fit(val_quantiles, val_targets)
    calib_test_quantiles = m.predict_interval(test_quantiles)
    check_quantiles(calib_test_quantiles)


def check_quantiles(quantiles: np.ndarray) -> None:
    assert quantiles.ndim == 2 and quantiles.shape[1] == 2
