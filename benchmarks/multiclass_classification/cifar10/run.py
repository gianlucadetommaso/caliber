import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from tabulate import tabulate

from caliber import (
    BrierLinearScalingMulticlassClassificationModel,
    CrossEntropyLinearScalingMulticlassClassificationModel,
    DistanceAwareExponentialInterpolantMulticlassClassificationModel,
    DistanceAwareKolmogorovInterpolantMulticlassClassificationModel,
    HistogramBinningMulticlassClassificationModel,
    KolmogorovInterpolantMulticlassClassificationModel,
)
from caliber.binary_classification.metrics import (
    false_negative_rate,
    false_positive_rate,
)
from caliber.multiclass_classification.metrics import (
    average_squared_calibration_error,
    brier_score_loss,
    expected_calibration_error,
)

SEED = 0
THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5
N_GROUPS = 5


data = np.load("cifar10_data.npy", allow_pickle=True).tolist()

calib_size = int(len(data["probs"]) * TRAIN_VAL_SPLIT)
calib_probs, test_probs = data["probs"][:calib_size], data["probs"][calib_size:]
calib_outputs, test_outputs = data["outputs"][:calib_size], data["outputs"][calib_size:]
calib_distances, test_distances = (
    data["distances"][:calib_size],
    data["distances"][calib_size:],
)
calib_targets, test_targets = data["targets"][:calib_size], data["targets"][calib_size:]
test_preds = np.argmax(test_probs, axis=1)

ood_data = np.load("cifar100_data.npy", allow_pickle=True).tolist()
ood_probs = ood_data["probs"]
ood_outputs = ood_data["outputs"]
ood_distances = ood_data["distances"]

inout_probs = np.concatenate((test_probs, ood_probs))
inout_targets = np.concatenate(
    (np.zeros(test_probs.shape[0]), np.ones(ood_probs.shape[0]))
)

models = {
    "histogram_binning": HistogramBinningMulticlassClassificationModel(),
    "cross_entropy_linear_scaling_shared": CrossEntropyLinearScalingMulticlassClassificationModel(
        has_shared_slope=True
    ),
    "brier_linear_scaling_shared": BrierLinearScalingMulticlassClassificationModel(
        has_shared_slope=True
    ),
    "cross_entropy_linear_scaling_unshared": CrossEntropyLinearScalingMulticlassClassificationModel(
        has_shared_slope=False
    ),
    "brier_linear_scaling_unshared": BrierLinearScalingMulticlassClassificationModel(
        has_shared_slope=False
    ),
    "dai_exp_ce_ls_unshared": DistanceAwareExponentialInterpolantMulticlassClassificationModel(
        CrossEntropyLinearScalingMulticlassClassificationModel()
    ),
    "dai_kolm_ce_ls_unshared": DistanceAwareKolmogorovInterpolantMulticlassClassificationModel(
        CrossEntropyLinearScalingMulticlassClassificationModel()
    ),
    "kolm_ce_ls_unshared": KolmogorovInterpolantMulticlassClassificationModel(
        CrossEntropyLinearScalingMulticlassClassificationModel()
    ),
}
performance_metrics = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
}
calibration_metrics = {
    "cross-entropy": log_loss,
    "Brier score": brier_score_loss,
    "ECE": expected_calibration_error,
    "ASCE": average_squared_calibration_error,
}

ood_metrics = {
    "ROCAUC": roc_auc_score,
    "PRAUC": average_precision_score,
    "FPR95": false_positive_rate,
    "FNR95": false_negative_rate,
}

results = {
    **{"uncalibrated": dict()},
    **{m_name: dict() for m_name, m in models.items()},
}

for metric_name, metric in performance_metrics.items():
    results["uncalibrated"][metric_name] = metric(test_targets, test_preds)
for metric_name, metric in calibration_metrics.items():
    results["uncalibrated"][metric_name] = metric(test_targets, test_probs)

inout_confs = inout_probs.max(1)
for metric_name, metric in ood_metrics.items():
    results["uncalibrated"][metric_name] = metric(
        inout_targets,
        (
            1 - inout_confs
            if metric.__name__ not in ["false_positive_rate", "false_negative_rate"]
            else inout_confs < 0.95
        ),
    )

for m_name, m in models.items():
    if m_name.startswith("dai"):
        m.fit(calib_probs, calib_distances, calib_targets)
        posthoc_test_probs = m.predict_proba(test_probs, test_distances)
        posthoc_test_preds = m.predict(test_probs, test_distances)
        posthoc_ood_probs = m.predict_proba(ood_probs, ood_distances)
    elif m_name.startswith("kolm"):
        m.fit(calib_probs, calib_outputs, calib_targets)
        posthoc_test_probs = m.predict_proba(test_probs, test_outputs)
        posthoc_test_preds = m.predict(test_probs, test_outputs)
        posthoc_ood_probs = m.predict_proba(ood_probs, ood_outputs)
    else:
        m.fit(calib_probs, calib_targets)
        posthoc_test_probs = m.predict_proba(test_probs)
        posthoc_test_preds = m.predict(test_probs)
        posthoc_ood_probs = m.predict_proba(ood_probs)
    posthoc_inout_confs = np.concatenate((posthoc_test_probs, posthoc_ood_probs)).max(1)
    for metric_name, metric in performance_metrics.items():
        results[m_name][metric_name] = metric(test_targets, posthoc_test_preds)
    for metric_name, metric in calibration_metrics.items():
        results[m_name][metric_name] = metric(test_targets, posthoc_test_probs)
    for metric_name, metric in ood_metrics.items():
        results[m_name][metric_name] = metric(
            inout_targets,
            (
                1 - posthoc_inout_confs
                if metric.__name__ not in ["false_positive_rate", "false_negative_rate"]
                else posthoc_inout_confs < 0.95
            ),
        )

print(
    tabulate(
        [[m] + list(r.values()) for m, r in results.items()],
        headers=[""] + list(results[list(results.keys())[0]].keys()),
        tablefmt="rounded_outline",
    ),
    "\n\n",
)
