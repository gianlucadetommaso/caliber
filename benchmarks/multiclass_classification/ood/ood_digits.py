import numpy as np
from sklearn import svm
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from tabulate import tabulate

from caliber import (
    CrossEntropyLinearScalingMulticlassClassificationModel,
    DistanceAwareExponentialInterpolantMulticlassClassificationModel,
    DistanceAwareHistogramBinningMulticlassClassificationModel,
    DistanceAwareKolmogorovInterpolantMulticlassClassificationModel,
    HistogramBinningMulticlassClassificationModel,
    KolmogorovInterpolantMulticlassClassificationModel,
)
from caliber.multiclass_classification.metrics import (
    brier_score_loss,
    expected_calibration_error,
)
from data import load_digits_data

SEED = 0
TRAIN_VAL_SPLIT = 0.5
N_OOD_SAMPLES = 1000


def distance_fn(inputs, _train_inputs):
    return np.min(
        np.mean((inputs[None] - _train_inputs[:, None]) ** 2, axis=-1), axis=0
    )


rng = np.random.default_rng(SEED)

train_inputs, test_inputs, train_targets, test_targets = load_digits_data()

lle = LocallyLinearEmbedding(method="modified", n_neighbors=64, n_components=32)
lle.fit(np.concatenate((train_inputs, test_inputs)))
train_embeddings = lle.transform(train_inputs)
test_embeddings = lle.transform(test_inputs)

train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_embeddings, val_embeddings = (
    train_embeddings[:train_size],
    train_embeddings[train_size:],
)
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]
ood_inputs = np.rot90(test_inputs.reshape(-1, 8, 8), axes=(1, 2)).reshape(-1, 64)
ood_embeddings = lle.transform(ood_inputs)

model = svm.SVC(gamma=0.001, probability=True, random_state=SEED)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)
test_probs = model.predict_proba(test_inputs)
test_preds = np.argmax(test_probs, 1)
ood_probs = model.predict_proba(ood_inputs)

val_distances = distance_fn(val_inputs, train_inputs)
test_distances = distance_fn(test_inputs, train_inputs)
ood_distances = distance_fn(ood_inputs, train_inputs)

inout_probs = np.max(np.concatenate((test_probs, ood_probs), axis=0), axis=1)
inout_targets = np.concatenate(
    (np.ones(test_probs.shape[0]), np.zeros(ood_probs.shape[0]))
)

results = dict()
results["uncalibrated"] = dict(
    model=model, test_probs=test_probs, test_preds=test_preds, inout_probs=inout_probs
)

dahb = DistanceAwareHistogramBinningMulticlassClassificationModel()
dahb.fit(val_probs, val_distances, val_targets)
calib_test_probs = dahb.predict_proba(test_probs, test_distances)
calib_test_preds = dahb.predict(test_probs, test_distances)
calib_ood_probs = dahb.predict_proba(ood_probs, ood_distances)
calib_inout_probs = np.max(
    np.concatenate((calib_test_probs, calib_ood_probs), axis=0), axis=1
)

results["DAHB"] = dict(
    model=dahb,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    inout_probs=calib_inout_probs,
)

hb = HistogramBinningMulticlassClassificationModel()
hb.fit(val_probs, val_targets)
calib_test_probs = hb.predict_proba(test_probs)
calib_test_preds = hb.predict(test_probs)
calib_ood_probs = hb.predict_proba(ood_probs)
calib_inout_probs = np.max(
    np.concatenate((calib_test_probs, calib_ood_probs), axis=0), axis=1
)

results["HB"] = dict(
    model=hb,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    inout_probs=calib_inout_probs,
)

ls = CrossEntropyLinearScalingMulticlassClassificationModel()
ls.fit(val_probs, val_targets)
calib_test_probs = ls.predict_proba(test_probs)
calib_test_preds = ls.predict(test_probs)
calib_ood_probs = ls.predict_proba(ood_probs)
calib_inout_probs = np.max(
    np.concatenate((calib_test_probs, calib_ood_probs), axis=0), axis=1
)

results["LS"] = dict(
    model=ls,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    inout_probs=calib_inout_probs,
)

daihb = DistanceAwareHistogramBinningMulticlassClassificationModel()
daihb.fit(val_probs, val_distances, val_targets)
calib_test_preds = daihb.predict(test_probs, test_distances)
calib_test_probs = daihb.predict_proba(test_probs, test_distances)
calib_ood_probs = daihb.predict_proba(ood_probs, ood_distances)
calib_inout_probs = np.max(
    np.concatenate((calib_test_probs, calib_ood_probs), axis=0), axis=1
)

results["DAIHB"] = dict(
    model=daihb,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    inout_probs=calib_inout_probs,
)

daiexpls = DistanceAwareExponentialInterpolantMulticlassClassificationModel(
    CrossEntropyLinearScalingMulticlassClassificationModel()
)
daiexpls.fit(val_probs, val_distances, val_targets)
calib_test_probs = daiexpls.predict_proba(test_probs, test_distances)
calib_test_preds = daiexpls.predict(test_probs, test_distances)
calib_ood_probs = daiexpls.predict_proba(ood_probs, ood_distances)
calib_inout_probs = np.max(
    np.concatenate((calib_test_probs, calib_ood_probs), axis=0), axis=1
)

results["DAIEXPLS"] = dict(
    model=daiexpls,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    inout_probs=calib_inout_probs,
)

daikolmls = DistanceAwareKolmogorovInterpolantMulticlassClassificationModel(
    CrossEntropyLinearScalingMulticlassClassificationModel()
)
daikolmls.fit(val_probs, val_distances, val_targets)
calib_test_probs = daikolmls.predict_proba(test_probs, test_distances)
calib_test_preds = daikolmls.predict(test_probs, test_distances)
calib_ood_probs = daikolmls.predict_proba(ood_probs, ood_distances)
calib_inout_probs = np.max(
    np.concatenate((calib_test_probs, calib_ood_probs), axis=0), axis=1
)

results["DAIKOLMLS"] = dict(
    model=daikolmls,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    inout_probs=calib_inout_probs,
)

kolmls = KolmogorovInterpolantMulticlassClassificationModel(
    CrossEntropyLinearScalingMulticlassClassificationModel()
)
kolmls.fit(val_probs, val_embeddings, val_targets)
calib_test_probs = kolmls.predict_proba(test_probs, test_embeddings)
calib_test_preds = kolmls.predict(test_probs, test_embeddings)
calib_ood_probs = kolmls.predict_proba(ood_probs, ood_embeddings)
calib_inout_probs = np.max(
    np.concatenate((calib_test_probs, calib_ood_probs), axis=0), axis=1
)

results["KOLMLS"] = dict(
    model=kolmls,
    test_probs=calib_test_probs,
    test_preds=calib_test_preds,
    inout_probs=calib_inout_probs,
)

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
