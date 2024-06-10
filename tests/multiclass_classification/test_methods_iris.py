import numpy as np
import pytest
from sklearn import svm

from caliber import (
    BrierLinearScalingMulticlassClassificationModel,
    CrossEntropyLinearScalingMulticlassClassificationModel,
    FocalLinearScalingMulticlassClassificationModel,
    HistogramBinningMulticlassClassificationModel,
)
from data import load_iris_data

SEED = 0
THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5
N_GROUPS = 5

METHODS = {
    "histogram_binning": HistogramBinningMulticlassClassificationModel(),
    "cross_entropy_linear_scaling_shared_slope_no_cross": CrossEntropyLinearScalingMulticlassClassificationModel(
        has_shared_slope=True,
        has_cross_slopes=False
    ),
    "brier_linear_scaling_shared_no_intercept": BrierLinearScalingMulticlassClassificationModel(
        has_intercept=False
    ),
    "focal_linear_scaling_shared_shared_intercept": FocalLinearScalingMulticlassClassificationModel(
        has_shared_intercept=True
    ),
    "cross_entropy_linear_scaling_unshared_no_cross": CrossEntropyLinearScalingMulticlassClassificationModel(
        has_cross_slopes=False
    ),
}

train_inputs, test_inputs, train_targets, test_targets = load_iris_data()

train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

model = svm.SVC(gamma=0.001, probability=True, random_state=SEED)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)
test_probs = model.predict_proba(test_inputs)
test_preds = np.argmax(test_probs, axis=1)


@pytest.mark.parametrize("m", list(METHODS.values()))
def test_method(m) -> None:
    m.fit(val_probs, val_targets)
    probs = m.predict_proba(test_probs)
    preds = m.predict(test_probs)
    check_probs_preds(probs, preds)


def check_probs_preds(probs: np.ndarray, preds: np.ndarray) -> None:
    assert probs.ndim == 2
    assert np.all(probs <= 1) and np.all(probs >= 0)
    assert preds.ndim == 1
    assert preds.dtype in ["int32", "int64"]
