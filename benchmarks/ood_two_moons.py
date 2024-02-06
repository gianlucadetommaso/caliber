import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate

from caliber import OODHistogramBinningBinaryClassificationModel
from caliber.binary_classification.data import load_two_moons_data

TRAIN_VAL_SPLIT = 0.5


def distance_fn(inputs, _train_inputs):
    return np.min(
        np.mean((inputs[None] - _train_inputs[:, None]) ** 2, axis=-1), axis=0
    )


train_inputs, test_inputs, train_targets, test_targets = load_two_moons_data()
train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]
ood_inputs = np.random.normal(loc=[1, 2], scale=0.31, size=(1000, 2))

model = MLPClassifier(random_state=43)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)[:, 1]
test_probs = model.predict_proba(test_inputs)[:, 1]
ood_probs = model.predict_proba(ood_inputs)[:, 1]

val_distances = distance_fn(val_inputs, train_inputs)
test_distances = distance_fn(test_inputs, train_inputs)
ood_distances = distance_fn(ood_inputs, train_inputs)

calib_model = OODHistogramBinningBinaryClassificationModel()
calib_model.fit(val_probs, val_distances, val_targets)
calib_test_probs = calib_model.predict_proba(test_probs, test_distances)
calib_ood_probs = calib_model.predict_proba(ood_probs, ood_distances)

grid_size = 150
xx = np.linspace(-2, 2.5, grid_size)
yy = np.linspace(-1, 2.5, grid_size)
grid_inputs = np.array([[_xx, _yy] for _xx in xx for _yy in yy])

grid_probs = model.predict_proba(grid_inputs)[:, 1]
grid_distances = distance_fn(grid_inputs, train_inputs)
calib_grid_probs = calib_model.predict_proba(grid_probs, grid_distances)

grid_confs = np.maximum(1 - grid_probs, grid_probs)
calib_grid_confs = np.maximum(1 - calib_grid_probs, calib_grid_probs)

fig = plt.figure(figsize=(6, 3))
grid_inputs = grid_inputs.reshape(grid_size, grid_size, 2)
for i, (k, c) in enumerate(
    dict(
        zip(["confidence before", "confidence after"], [grid_confs, calib_grid_confs])
    ).items()
):
    plt.subplot(1, 2, i + 1)
    plt.title(k, fontsize=12)
    im = plt.pcolor(
        grid_inputs[:, :, 0], grid_inputs[:, :, 1], c.reshape(grid_size, grid_size)
    )
    plt.scatter(
        *test_inputs.T,
        s=1,
        c=["C0" if i == 1 else "C1" for i in test_targets],
    )
    plt.colorbar(im)
plt.tight_layout()
plt.show()


inout_probs = np.concatenate((test_probs, ood_probs))
inout_targets = np.concatenate((np.zeros_like(test_probs), np.ones_like(ood_probs)))
calib_inout_probs = np.concatenate((calib_test_probs, calib_ood_probs))
print(
    tabulate(
        [
            [
                "before",
                roc_auc_score(inout_targets, inout_probs),
                average_precision_score(inout_targets, inout_probs),
            ],
            [
                "after",
                roc_auc_score(inout_targets, calib_inout_probs),
                average_precision_score(inout_targets, calib_inout_probs),
            ],
        ],
        headers=["", "ROC AUC score", "avg. prec. score"],
        tablefmt="rounded_outline",
    )
)
