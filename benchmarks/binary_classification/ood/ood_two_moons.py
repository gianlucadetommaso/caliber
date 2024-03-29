import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate

from caliber import (
    DistanceAwareExponentialInterpolantBinaryClassificationModel,
    DistanceAwareHistogramBinningBinaryClassificationModel,
    DistanceAwareKolmogorovInterpolantBinaryClassificationModel,
    HistogramBinningBinaryClassificationModel,
    MahalanobisBinaryClassificationModel,
)
from caliber.binary_classification.metrics import expected_calibration_error
from data import load_two_moons_data

TRAIN_VAL_SPLIT = 0.5


def distance_fn(inputs, _train_inputs):
    return np.min(
        np.mean((inputs[None] - _train_inputs[:, None]) ** 2, axis=-1), axis=0
    )


train_inputs, test_inputs, train_targets, test_targets = load_two_moons_data()
train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]
ood_inputs = np.random.normal(loc=[1, 1], scale=0.31, size=(1000, 2))

model = MLPClassifier(random_state=43)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)[:, 1]
test_probs = model.predict_proba(test_inputs)[:, 1]
test_preds = model.predict(test_inputs)
ood_probs = model.predict_proba(ood_inputs)[:, 1]

inout_probs = np.concatenate((test_probs, ood_probs))
inout_targets = np.concatenate((np.zeros_like(test_probs), np.ones_like(ood_probs)))

results = dict()
results["uncalibrated"] = dict(
    model=model,
    test_probs=test_probs,
    test_preds=test_preds,
    ood_probs=ood_probs,
    inout_probs=inout_probs,
)

val_distances = distance_fn(val_inputs, train_inputs)
test_distances = distance_fn(test_inputs, train_inputs)
ood_distances = distance_fn(ood_inputs, train_inputs)

dahb = DistanceAwareHistogramBinningBinaryClassificationModel()
dahb.fit(val_probs, val_distances, val_targets)
calib_test_probs = dahb.predict_proba(test_probs, test_distances)
calib_test_preds = dahb.predict(test_probs, test_distances)
calib_ood_probs = dahb.predict_proba(ood_probs, ood_distances)
calib_inout_probs = np.concatenate((calib_test_probs, calib_ood_probs))

results["DAHB"] = dict(
    model=dahb,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    ood_probs=calib_ood_probs,
    inout_probs=calib_inout_probs,
)

hb = HistogramBinningBinaryClassificationModel()
hb.fit(val_probs, val_targets)
calib_test_probs = hb.predict_proba(test_probs)
calib_test_preds = hb.predict(test_probs)
calib_ood_probs = hb.predict_proba(ood_probs)
calib_inout_probs = np.concatenate((calib_test_probs, calib_ood_probs))

results["HB"] = dict(
    model=hb,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    ood_probs=calib_ood_probs,
    inout_probs=calib_inout_probs,
)

daiexp = DistanceAwareExponentialInterpolantBinaryClassificationModel(
    HistogramBinningBinaryClassificationModel()
)
daiexp.fit(val_probs, val_distances, val_targets)
calib_test_probs = daiexp.predict_proba(test_probs, test_distances)
calib_test_preds = daiexp.predict(test_probs, test_distances)
calib_ood_probs = daiexp.predict_proba(ood_probs, ood_distances)
calib_inout_probs = np.concatenate((calib_test_probs, calib_ood_probs))

results["DAIEXP"] = dict(
    model=daiexp,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    ood_probs=calib_ood_probs,
    inout_probs=calib_inout_probs,
)

daikol = DistanceAwareKolmogorovInterpolantBinaryClassificationModel(
    HistogramBinningBinaryClassificationModel()
)
daikol.fit(val_probs, val_distances, val_targets)
calib_test_probs = daikol.predict_proba(test_probs, test_distances)
calib_test_preds = daikol.predict(test_probs, test_distances)
calib_ood_probs = daikol.predict_proba(ood_probs, ood_distances)
calib_inout_probs = np.concatenate((calib_test_probs, calib_ood_probs))

results["DAIKOL"] = dict(
    model=daikol,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    ood_probs=calib_ood_probs,
    inout_probs=calib_inout_probs,
)

maha = MahalanobisBinaryClassificationModel(threshold=0.5)
maha.fit(val_inputs, val_targets)
calib_test_probs = maha.predict_proba(test_inputs)
calib_test_preds = maha.predict(test_inputs)
calib_ood_probs = maha.predict_proba(ood_inputs)
calib_inout_probs = np.concatenate((calib_test_probs, calib_ood_probs))

results["MAHA_WITH"] = dict(
    model=maha,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    ood_probs=calib_ood_probs,
    inout_probs=calib_inout_probs,
)

maha = MahalanobisBinaryClassificationModel(threshold=0.5)
maha.fit(val_inputs)
calib_test_probs = maha.predict_proba(test_inputs)
calib_test_preds = maha.predict(test_inputs)
calib_ood_probs = maha.predict_proba(ood_inputs)
calib_inout_probs = np.concatenate((calib_test_probs, calib_ood_probs))

results["MAHA_WITHOUT"] = dict(
    model=maha,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    ood_probs=calib_ood_probs,
    inout_probs=calib_inout_probs,
)

grid_size = 150
xx = np.linspace(-2, 2.5, grid_size)
yy = np.linspace(-1, 2.5, grid_size)
grid_inputs = np.array([[_xx, _yy] for _xx in xx for _yy in yy])
grid_inputs2 = grid_inputs.reshape(grid_size, grid_size, 2)

grid_probs = model.predict_proba(grid_inputs)[:, 1]
grid_distances = distance_fn(grid_inputs, train_inputs)
grid_confs = np.maximum(1 - grid_probs, grid_probs)

fig, axes = plt.subplots(
    nrows=len(results) - 1, ncols=2, figsize=(6, 1.5 * (len(results) - 1))
)
for i, k in enumerate({k: v for k, v in results.items() if k != "uncalibrated"}):
    if k in ["DAHB", "DAIEXP", "DAIKOL", "MAHA_WITH", "MAHA_WITHOUT"]:
        if k.startswith("MAHA"):
            calib_grid_probs = results[k]["model"].predict_proba(grid_inputs)
        else:
            calib_grid_probs = results[k]["model"].predict_proba(
                grid_probs, grid_distances
            )
    else:
        calib_grid_probs = results[k]["model"].predict_proba(grid_probs)

    calib_grid_confs = np.maximum(1 - calib_grid_probs, calib_grid_probs)
    for j, (s, c) in enumerate(
        dict(zip(["uncalibrated", k], [grid_confs, calib_grid_confs])).items()
    ):
        axes[i, 1].set_title(s, fontsize=12)
        im = axes[i, j].pcolor(
            grid_inputs2[:, :, 0],
            grid_inputs2[:, :, 1],
            c.reshape(grid_size, grid_size),
        )
        axes[i, j].scatter(
            *test_inputs.T,
            s=1,
            c=["C0" if i == 1 else "C1" for i in test_targets],
        )
        plt.colorbar(im, ax=axes[i, j])
plt.tight_layout()
plt.show()

print(
    tabulate(
        [
            [
                k,
                roc_auc_score(inout_targets, v["inout_probs"]),
                average_precision_score(inout_targets, v["inout_probs"]),
                accuracy_score(test_targets, v["test_preds"]),
                brier_score_loss(test_targets, v["test_probs"]),
                expected_calibration_error(test_targets, v["test_probs"]),
            ]
            for k, v in results.items()
        ],
        headers=[
            "",
            "ROC AUC score",
            "avg. prec. score",
            "accuracy",
            "Brier score",
            "ECE",
        ],
        tablefmt="rounded_outline",
    )
)
