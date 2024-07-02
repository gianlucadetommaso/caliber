from caliber import HeteroskedasticLinearRegressionModel
from data import load_regression_data
import numpy as np
from typing import Tuple


CONFIDENCE = 0.95
TRAIN_VAL_SPLIT = 0.5


train_inputs, test_inputs, train_targets, test_targets = load_regression_data()
train_targets -= train_targets.mean()
train_targets /= train_targets.std()
test_targets -= test_targets.mean()
test_targets /= test_targets.std()
train_inputs -= train_inputs.mean()
train_inputs /= train_inputs.std()
test_inputs -= test_inputs.mean()
test_inputs /= test_inputs.std()


def mean_logstd_predict_fn(params: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return params[0] + np.dot(features, params[1:3]), params[3] + np.dot(features, params[4:6])


def test_heteroskedastic_regression_model() -> None:
    model = HeteroskedasticLinearRegressionModel(
        minimize_options=dict(x0=np.array([0, 1, 1, 0, 1, 1]))
    )
    model.fit(train_inputs, train_targets)
    means = model.predict(test_inputs)
    stds = model.predict_std(test_inputs)
    vars = model.predict_var(test_inputs)
    quantiles = model.predict_quantile(test_inputs, 0.95)

    test_size = len(test_targets)
    assert len(means) == test_size
    assert len(stds) == test_size
    assert len(vars) == test_size
    assert len(quantiles) == test_size
    assert np.all(stds >= 0)
    assert np.all(vars >= 0)
