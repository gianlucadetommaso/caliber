from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
import numpy as np
from tabulate import tabulate
from xgboost import XGBClassifier

from caliber import (
    HistogramBinningBinaryClassificationModel,
    PPIHistogramBinningBinaryClassificationModel,
)
from caliber.binary_classification.metrics import (
    average_smooth_squared_calibration_error,
    average_squared_calibration_error,
    expected_calibration_error,
)
from data import load_two_moons_data

THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5
N_GROUPS = 5
PSEUDO_FRAC = 0.2


datasets = {
    "two_moons": load_two_moons_data(),
}

for dataset_name, dataset in datasets.items():
    train_inputs, test_inputs, train_targets, test_targets = dataset

    train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
    train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
    train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

    model = XGBClassifier(random_state=42)
    model.fit(train_inputs, train_targets)

    val_probs = model.predict_proba(val_inputs)[:, 1]
    test_probs = model.predict_proba(test_inputs)[:, 1]
    test_preds = (test_probs >= THRESHOLD).astype(int)

    # use probs as pseudo targets
    val_pseudo_size = int(np.ceil(len(val_targets) * PSEUDO_FRAC))
    val_pseudo_indices = np.arange(val_pseudo_size)
    val_non_pseudo_probs = val_probs[val_pseudo_size:]
    val_non_pseudo_targets = val_targets[val_pseudo_size:]
    val_targets = np.concatenate((val_probs[:val_pseudo_size], val_non_pseudo_targets))

    posthoc_models = {
        "hb": HistogramBinningBinaryClassificationModel(),
        "ppi_hb": PPIHistogramBinningBinaryClassificationModel(),
    }
    performance_metrics = {
        # "accuracy": accuracy_score,
        # "balanced_accuracy": balanced_accuracy_score,
        # "precision": precision_score,
        # "recall": recall_score,
        # "positive_F1": f1_score,
    }
    calibration_metrics = {
        "cross-entropy": log_loss,
        "Brier score": brier_score_loss,
        "ASCE": average_squared_calibration_error,
        # "ASSCE": average_smooth_squared_calibration_error,
        "ECE": expected_calibration_error,
    }

    results = {
        **{model.__class__.__name__: dict()},
        **{m_name: dict() for m_name, m in posthoc_models.items()},
    }

    for metric_name, metric in performance_metrics.items():
        results[model.__class__.__name__][metric_name] = metric(
            test_targets, test_preds
        )
    for metric_name, metric in calibration_metrics.items():
        results[model.__class__.__name__][metric_name] = metric(
            test_targets, test_probs
        )

    for m_name, m in posthoc_models.items():
        if "ppi" not in m_name:
            m.fit(val_non_pseudo_probs, val_non_pseudo_targets)
            posthoc_test_probs = m.predict_proba(test_probs)
            posthoc_test_preds = m.predict(test_probs)
        else:
            m.fit(val_probs, val_targets, val_pseudo_indices)
            posthoc_test_probs = m.predict_proba(test_probs)
            posthoc_test_preds = m.predict(test_probs)
        for metric_name, metric in performance_metrics.items():
            results[m_name][metric_name] = metric(test_targets, posthoc_test_preds)
        for metric_name, metric in calibration_metrics.items():
            results[m_name][metric_name] = metric(test_targets, posthoc_test_probs)

    print(
        tabulate(
            [[m] + list(r.values()) for m, r in results.items()],
            headers=[dataset_name.upper()]
            + list(results[list(results.keys())[0]].keys()),
            tablefmt="rounded_outline",
        ),
        "\n\n",
    )
