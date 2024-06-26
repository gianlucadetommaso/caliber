import numpy as np
import pytest
from sklearn import svm

from caliber import (
    DistanceAwareExponentialInterpolantMulticlassClassificationModel,
    DistanceAwareHistogramBinningMulticlassClassificationModel,
    DistanceAwareKolmogorovInterpolantMulticlassClassificationModel,
)
from data import load_digits_data

SEED = 0
TRAIN_VAL_SPLIT = 0.5

METHODS = {
    "da_hb": DistanceAwareHistogramBinningMulticlassClassificationModel(),
    "da_exponential": DistanceAwareExponentialInterpolantMulticlassClassificationModel(),
    "da_kolmogorov": DistanceAwareKolmogorovInterpolantMulticlassClassificationModel(),
}


def distance_fn(inputs, _train_inputs):
    return np.min(
        np.mean((inputs[None] - _train_inputs[:, None]) ** 2, axis=-1), axis=0
    )


rng = np.random.default_rng(SEED)

train_inputs, test_inputs, train_targets, test_targets = load_digits_data()
train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

model = svm.SVC(gamma=0.001, probability=True)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)
test_probs = model.predict_proba(test_inputs)

val_distances = distance_fn(val_inputs, train_inputs)
test_distances = distance_fn(test_inputs, train_inputs)


@pytest.mark.parametrize("m", list(METHODS.values()))
def test_method(m) -> None:
    m.fit(val_probs, val_distances, val_targets)
    calib_test_probs = m.predict_proba(test_probs, test_distances)
    calib_test_preds = m.predict(test_probs, test_distances)
    check_probs_preds(calib_test_probs, calib_test_preds)


def check_probs_preds(probs: np.ndarray, preds: np.ndarray) -> None:
    assert probs.ndim == 2
    assert np.all(probs <= 1) and np.all(probs >= 0)
    assert preds.ndim == 1
    assert set(preds).issubset(set(np.arange(probs.shape[1])))
