import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tabulate import tabulate

from caliber import MahalanobisBinaryClassificationModel
from data import load_two_moons_data

train_inputs, test_inputs, train_targets, test_targets = load_two_moons_data()
ood_inputs = np.random.normal(loc=[1, 1], scale=0.31, size=(1000, 2))
inout_targets = np.concatenate((np.zeros(len(test_inputs)), np.ones(len(ood_inputs))))

maha = MahalanobisBinaryClassificationModel(threshold=0.5)
maha.fit(train_inputs, train_targets)
test_probs = maha.predict_proba(test_inputs)
ood_probs = maha.predict_proba(ood_inputs)
inout_probs = np.concatenate((test_probs, ood_probs))

results = dict()
results["mahalanobis_with_targets"] = dict(
    model=maha,
    inout_probs=inout_probs,
)

maha = MahalanobisBinaryClassificationModel(threshold=0.5)
maha.fit(train_inputs)
test_probs = maha.predict_proba(test_inputs)
test_preds = maha.predict(test_inputs)
ood_probs = maha.predict_proba(ood_inputs)
ood_preds = maha.predict(ood_inputs)
inout_probs = np.concatenate((test_probs, ood_probs))

results["mahalanobis_without_targets"] = dict(
    model=maha,
    inout_probs=inout_probs,
)

grid_size = 150
xx = np.linspace(-2, 2.5, grid_size)
yy = np.linspace(-1, 1.5, grid_size)
grid_inputs = np.array([[_xx, _yy] for _xx in xx for _yy in yy])
grid_inputs2 = grid_inputs.reshape(grid_size, grid_size, 2)

fig, axes = plt.subplots(nrows=1, ncols=len(results), figsize=(4 * len(results), 3))
for i, (k, v) in enumerate(results.items()):
    grid_probs = v["model"].predict_proba(grid_inputs)
    grid_preds = v["model"].predict(grid_inputs)

    im = axes[i].pcolor(
        grid_inputs2[:, :, 0],
        grid_inputs2[:, :, 1],
        grid_probs.reshape(grid_size, grid_size),
    )
    plt.colorbar(im, ax=axes[i])
    axes[i].scatter(
        *test_inputs.T,
        s=1,
        c="C0",
    )
plt.tight_layout()
plt.show()

print(
    tabulate(
        [
            [
                k,
                roc_auc_score(inout_targets, v["inout_probs"]),
                average_precision_score(inout_targets, v["inout_probs"]),
            ]
            for k, v in results.items()
        ],
        headers=[
            "",
            "ROC AUC score",
            "avg. prec. score",
        ],
        tablefmt="rounded_outline",
    )
)
