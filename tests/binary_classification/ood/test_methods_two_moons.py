import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from caliber import (
    DistanceAwareExponentialInterpolantBinaryClassificationModel,
    DistanceAwareHistogramBinningBinaryClassificationModel,
    DistanceAwareKolmogorovInterpolantBinaryClassificationModel,
)
from data import load_two_moons_data

TRAIN_VAL_SPLIT = 0.5

METHODS = {
    "da_hb": DistanceAwareHistogramBinningBinaryClassificationModel(),
    "da_exponential": DistanceAwareExponentialInterpolantBinaryClassificationModel(),
    "da_kolmogorov": DistanceAwareKolmogorovInterpolantBinaryClassificationModel(),
}


def distance_fn(inputs, _train_inputs):
    return np.min(
        np.mean((inputs[None] - _train_inputs[:, None]) ** 2, axis=-1), axis=0
    )


train_inputs, test_inputs, train_targets, test_targets = load_two_moons_data()
train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

model = MLPClassifier(random_state=43)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)[:, 1]
test_probs = model.predict_proba(test_inputs)[:, 1]

val_distances = distance_fn(val_inputs, train_inputs)
test_distances = distance_fn(test_inputs, train_inputs)


@pytest.mark.parametrize("method_name", list(METHODS.keys()))
def test_method(method_name: str) -> None:
    m = METHODS[method_name]
    m.fit(val_probs, val_distances, val_targets)
    calib_test_probs = m.predict_proba(test_probs, test_distances)
    calib_test_preds = m.predict(test_probs, test_distances)
    check_probs_preds(calib_test_probs, calib_test_preds)


def check_probs_preds(probs: np.ndarray, preds: np.ndarray) -> None:
    assert probs.ndim == 1
    assert np.all(probs <= 1) and np.all(probs >= 0)
    assert preds.ndim == 1
    assert set(preds) in [{0, 1}, {0}, {1}]
