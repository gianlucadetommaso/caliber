import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from tabulate import tabulate

from caliber import (
    BrierMulticlassClassificationLinearScaling,
    CrossEntropyMulticlassClassificationLinearScaling,
    HistogramBinningMulticlassClassificationModel,
)
from caliber.multiclass_classification.metrics import (
    average_squared_calibration_error,
    brier_score_loss,
    expected_calibration_error,
)
from data import load_iris_data

SEED = 0
THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5
N_GROUPS = 5


datasets = {
    "iris": load_iris_data(),
}


for dataset_name, dataset in datasets.items():
    train_inputs, test_inputs, train_targets, test_targets = dataset

    train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
    train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
    train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

    model = svm.SVC(gamma=0.001, probability=True, random_state=SEED)
    model.fit(train_inputs, train_targets)

    val_probs = model.predict_proba(val_inputs)
    test_probs = model.predict_proba(test_inputs)
    test_preds = np.argmax(test_probs, axis=1)

    posthoc_models = {
        "histogram_binning": HistogramBinningMulticlassClassificationModel(),
        "cross_entropy_linear_scaling_shared": CrossEntropyMulticlassClassificationLinearScaling(
            has_shared_slope=True
        ),
        "brier_linear_scaling_shared": BrierMulticlassClassificationLinearScaling(
            has_shared_slope=True
        ),
        "cross_entropy_linear_scaling_unshared": CrossEntropyMulticlassClassificationLinearScaling(
            has_shared_slope=False
        ),
        "brier_linear_scaling_unshared": BrierMulticlassClassificationLinearScaling(
            has_shared_slope=False
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
            headers=[dataset_name.upper()]
            + list(results[list(results.keys())[0]].keys()),
            tablefmt="rounded_outline",
        ),
        "\n\n",
    )
