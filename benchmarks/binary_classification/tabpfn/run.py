import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from tabpfn import TabPFNClassifier
from tabulate import tabulate
from xgboost import XGBClassifier

from caliber import (
    BetaBinaryClassificationModel,
    BrierLinearScalingBinaryClassificationModel,
    HistogramBinningBinaryClassificationModel,
    IsotonicRegressionBinaryClassificationModel,
    IterativeBinningBinaryClassificationModel,
)
from caliber.binary_classification.metrics import (
    average_squared_calibration_error,
    expected_calibration_error,
)
from data import load_breast_cancer_data

THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5

train_inputs, test_inputs, train_targets, test_targets = load_breast_cancer_data()

train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

xgboost_model = XGBClassifier(objective="binary:logistic")
xgboost_model.fit(train_inputs, train_targets)
test_xgboost_probs = xgboost_model.predict_proba(test_inputs)[:, 1]
test_xgboost_preds = (test_xgboost_probs >= THRESHOLD).astype(int)

tabpfn_model = TabPFNClassifier()
tabpfn_model.fit(train_inputs, train_targets)

val_probs = tabpfn_model.predict_proba(val_inputs)[:, 1]
test_probs = tabpfn_model.predict_proba(test_inputs)[:, 1]
test_preds = (test_probs >= THRESHOLD).astype(int)

posthoc_models = {
    "beta": BetaBinaryClassificationModel(),
    "histogram_binning": HistogramBinningBinaryClassificationModel(),
    "isotonic_regression": IsotonicRegressionBinaryClassificationModel(),
    "iterative_linear_binning": IterativeBinningBinaryClassificationModel(
        bin_model=BrierLinearScalingBinaryClassificationModel(),
    ),
}
performance_metrics = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "positive_F1": f1_score,
}
calibration_metrics = {
    "cross-entropy": log_loss,
    "Brier score": brier_score_loss,
    "ASCE": average_squared_calibration_error,
    "ECE": expected_calibration_error,
}

results = {
    **{xgboost_model.__class__.__name__: dict()},
    **{tabpfn_model.__class__.__name__: dict()},
    **{m_name: dict() for m_name, m in posthoc_models.items()},
}

for metric_name, metric in performance_metrics.items():
    results[xgboost_model.__class__.__name__][metric_name] = metric(
        test_targets, test_xgboost_preds
    )
    results[tabpfn_model.__class__.__name__][metric_name] = metric(
        test_targets, test_preds
    )
for metric_name, metric in calibration_metrics.items():
    results[xgboost_model.__class__.__name__][metric_name] = metric(
        test_targets, test_xgboost_probs
    )
    results[tabpfn_model.__class__.__name__][metric_name] = metric(
        test_targets, test_probs
    )

for m_name, m in posthoc_models.items():
    m.fit(val_probs, val_targets)
    posthoc_test_probs = m.predict_proba(test_probs)
    posthoc_test_preds = m.predict(test_probs)

    for metric_name, metric in performance_metrics.items():
        results[m_name][metric_name] = metric(test_targets, posthoc_test_preds)
    for metric_name, metric in calibration_metrics.items():
        results[m_name][metric_name] = metric(test_targets, posthoc_test_probs)

print(
    tabulate(
        [[m] + list(r.values()) for m, r in results.items()],
        headers=["BREAST CANCER"] + list(results[list(results.keys())[0]].keys()),
        tablefmt="rounded_outline",
    ),
    "\n\n",
)
