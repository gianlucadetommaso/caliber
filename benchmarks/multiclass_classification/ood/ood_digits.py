import numpy as np
from sklearn import svm
from tabulate import tabulate

from caliber import OODHistogramBinningMulticlassClassificationModel
from caliber.binary_classification.data import load_digits_data
from caliber.multiclass_classification.metrics import (
    brier_score,
    expected_calibration_error,
)

SEED = 0
TRAIN_VAL_SPLIT = 0.5
N_OOD_SAMPLES = 1000


def distance_fn(inputs, _train_inputs):
    return np.min(
        np.mean((inputs[None] - _train_inputs[:, None]) ** 2, axis=-1), axis=0
    )


rng = np.random.default_rng(SEED)

train_inputs, test_inputs, train_targets, test_targets = load_digits_data()
train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]
ood_inputs = rng.normal(size=train_inputs.shape)

model = svm.SVC(gamma=0.001, probability=True)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)
test_probs = model.predict_proba(test_inputs)
ood_probs = model.predict_proba(ood_inputs)

val_distances = distance_fn(val_inputs, train_inputs)
test_distances = distance_fn(test_inputs, train_inputs)
ood_distances = distance_fn(ood_inputs, train_inputs)

calib_model = OODHistogramBinningMulticlassClassificationModel()
calib_model.fit(val_probs, val_distances, val_targets)
calib_test_probs = calib_model.predict_proba(test_probs, test_distances)
calib_ood_probs = calib_model.predict_proba(ood_probs, ood_distances)

print(
    tabulate(
        [
            [
                "before",
                expected_calibration_error(test_targets, test_probs),
                brier_score(test_targets, test_probs),
            ],
            [
                "after",
                expected_calibration_error(test_targets, calib_test_probs),
                brier_score(test_targets, calib_test_probs),
            ],
        ],
        headers=["", "ECE", "Brier score"],
        tablefmt="rounded_outline",
    )
)
