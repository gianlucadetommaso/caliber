from sklearn import svm
from caliber.binary_classification.data import load_digits_data
from caliber import OODHistogramBinningMulticlassClassificationModel
import numpy as np


SEED = 0
TRAIN_VAL_SPLIT = 0.5


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


def test_method():
    calib_model = OODHistogramBinningMulticlassClassificationModel()
    calib_model.fit(val_probs, val_distances, val_targets)
    calib_test_probs = calib_model.predict_proba(test_probs, test_distances)
    calib_test_preds = calib_model.predict(test_probs, test_distances)
    check_probs_preds(calib_test_probs, calib_test_preds)


def check_probs_preds(probs: np.ndarray, preds: np.ndarray):
    assert probs.ndim == 2
    assert np.all(probs <= 1) and np.all(probs >= 0)
    assert preds.ndim == 1
    assert set(preds) == set(np.arange(probs.shape[1]))
