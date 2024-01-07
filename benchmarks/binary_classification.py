from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate

from caliber import (
    ASCEBinaryClassificationLinearScaling,
    BalancedAccuracyBinaryClassificationLinearScaling,
    BrierBinaryClassificationLinearScaling,
    CrossEntropyBinaryClassificationLinearScaling,
    ECEBinaryClassificationLinearScaling,
    NegativeF1BinaryClassificationLinearScaling,
    PositiveF1BinaryClassificationLinearScaling,
    PositiveNegativeRatesBinaryClassificationLinearScaling,
    PredictiveValuesBinaryClassificationLinearScaling,
    RighteousnessBinaryClassificationLinearScaling,
)
from caliber.binary_classification.metrics import (
    average_squared_calibration_error,
    expected_calibration_error,
)

THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5


def load_two_moons_data(
    n_train_samples=1000, n_test_samples=1000, noise=0.1, random_state=0
):
    _train_inputs, _train_targets = make_moons(
        n_samples=n_train_samples, noise=noise, random_state=random_state
    )
    _test_inputs, _test_targets = make_moons(
        n_samples=n_test_samples, noise=noise, random_state=random_state + 1
    )
    return _train_inputs, _test_inputs, _train_targets, _test_targets


def load_breast_cancer_data(test_size=0.1, random_state=0):
    data = load_breast_cancer()
    inputs = data.data
    targets = data.target
    _train_inputs, _test_inputs, _train_targets, _test_targets = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )
    return _train_inputs, _test_inputs, _train_targets, _test_targets


datasets = {
    "two_moons": load_two_moons_data(),
    "breast_cancer": load_breast_cancer_data(),
}

for dataset_name, dataset in datasets.items():
    train_inputs, test_inputs, train_targets, test_targets = dataset
    train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
    train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
    train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

    model = MLPClassifier(random_state=42)
    model.fit(train_inputs, train_targets)

    val_probs = model.predict_proba(val_inputs)[:, 1]
    test_probs = model.predict_proba(test_inputs)[:, 1]
    test_preds = (test_probs >= THRESHOLD).astype(int)

    posthoc_models = [
        BalancedAccuracyBinaryClassificationLinearScaling(threshold=THRESHOLD),
        PositiveF1BinaryClassificationLinearScaling(threshold=THRESHOLD),
        NegativeF1BinaryClassificationLinearScaling(threshold=THRESHOLD),
        PredictiveValuesBinaryClassificationLinearScaling(threshold=THRESHOLD),
        PositiveNegativeRatesBinaryClassificationLinearScaling(threshold=THRESHOLD),
        RighteousnessBinaryClassificationLinearScaling(threshold=THRESHOLD),
        BrierBinaryClassificationLinearScaling(),
        CrossEntropyBinaryClassificationLinearScaling(),
        ASCEBinaryClassificationLinearScaling(),
        ECEBinaryClassificationLinearScaling(),
    ]
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
        **{model.__class__.__name__: dict()},
        **{m.__class__.__name__: dict() for m in posthoc_models},
    }

    for metric_name, metric in performance_metrics.items():
        results[model.__class__.__name__][metric_name] = metric(
            test_targets, test_preds
        )
    for metric_name, metric in calibration_metrics.items():
        results[model.__class__.__name__][metric_name] = metric(
            test_targets, test_probs
        )

    for m in posthoc_models:
        m.fit(val_probs, val_targets)
        posthoc_test_probs = m.predict_proba(test_probs)
        posthoc_test_preds = m.predict(test_probs)
        for metric_name, metric in performance_metrics.items():
            results[m.__class__.__name__][metric_name] = metric(
                test_targets, posthoc_test_preds
            )
        for metric_name, metric in calibration_metrics.items():
            results[m.__class__.__name__][metric_name] = metric(
                test_targets, posthoc_test_probs
            )

    print(
        tabulate(
            [[m] + list(r.values()) for m, r in results.items()],
            headers=[dataset_name.upper()]
            + list(results[list(results.keys())[0]].keys()),
            tablefmt="rounded_outline",
        ),
        "\n\n",
    )
