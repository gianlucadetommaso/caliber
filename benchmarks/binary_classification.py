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
    HistogramBinningBinaryClassificationModel,
    NegativeF1BinaryClassificationLinearScaling,
    PositiveF1BinaryClassificationLinearScaling,
    PositiveNegativeRatesBinaryClassificationLinearScaling,
    PredictiveValuesBinaryClassificationLinearScaling,
    RighteousnessBinaryClassificationLinearScaling,
    IsotonicRegressionBinaryClassificationModel
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

    posthoc_models = {
        "balanced_accuracy_linear_scaling": BalancedAccuracyBinaryClassificationLinearScaling(
            threshold=THRESHOLD
        ),
        "balanced_accuracy_temperature_scaling": BalancedAccuracyBinaryClassificationLinearScaling(
            threshold=THRESHOLD, has_intercept=False
        ),
        "positive_f1_linear_scaling": PositiveF1BinaryClassificationLinearScaling(
            threshold=THRESHOLD
        ),
        "positive_f1_temperature_scaling": PositiveF1BinaryClassificationLinearScaling(
            threshold=THRESHOLD, has_intercept=False
        ),
        "negative_f1_linear_scaling": NegativeF1BinaryClassificationLinearScaling(
            threshold=THRESHOLD
        ),
        "negative_f1_temperature_scaling": NegativeF1BinaryClassificationLinearScaling(
            threshold=THRESHOLD, has_intercept=False
        ),
        "predictive_values_linear_scaling": PredictiveValuesBinaryClassificationLinearScaling(
            threshold=THRESHOLD
        ),
        "predictive_values_temperature_scaling": PredictiveValuesBinaryClassificationLinearScaling(
            threshold=THRESHOLD, has_intercept=False
        ),
        "positive_negative_rates_linear_scaling": PositiveNegativeRatesBinaryClassificationLinearScaling(
            threshold=THRESHOLD
        ),
        "positive_negative_rates_temperature_scaling": PositiveNegativeRatesBinaryClassificationLinearScaling(
            threshold=THRESHOLD, has_intercept=False
        ),
        "righteousness_linear_scaling": RighteousnessBinaryClassificationLinearScaling(
            threshold=THRESHOLD
        ),
        "righteousness_temperature_scaling": RighteousnessBinaryClassificationLinearScaling(
            threshold=THRESHOLD, has_intercept=False
        ),
        "brier_linear_scaling": BrierBinaryClassificationLinearScaling(),
        "brier_temperature_scaling": BrierBinaryClassificationLinearScaling(
            has_intercept=False
        ),
        "cross_entropy_linear_scaling": CrossEntropyBinaryClassificationLinearScaling(),
        "cross_entropy_temperature_scaling": CrossEntropyBinaryClassificationLinearScaling(
            has_intercept=False
        ),
        "asce_linear_scaling": ASCEBinaryClassificationLinearScaling(),
        "asce_temperature_scaling": ASCEBinaryClassificationLinearScaling(
            has_intercept=False
        ),
        "ece_linear_scaling": ECEBinaryClassificationLinearScaling(),
        "ece_temperature_scaling": ECEBinaryClassificationLinearScaling(
            has_intercept=False
        ),
        "histogram_binning": HistogramBinningBinaryClassificationModel(),
        "isotonic_regression": IsotonicRegressionBinaryClassificationModel()
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
