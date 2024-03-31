import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture
from tabulate import tabulate
from xgboost import XGBClassifier

from caliber import (
    BalancedAccuracyLinearScalingBinaryClassificationModel,
    BrierLinearScalingBinaryClassificationModel,
    CrossEntropyLinearScalingBinaryClassificationModel,
    FocalLinearScalingBinaryClassificationModel,
    GroupConditionalUnbiasedBinaryClassificationModel,
    HistogramBinningBinaryClassificationModel,
    IsotonicRegressionBinaryClassificationModel,
    IterativeBinningBinaryClassificationModel,
    IterativeFittingBinaryClassificationModel,
    IterativeSmoothHistogramBinningBinaryClassificationModel,
    ModelBiasBinaryClassificationConstantShift,
    NegativeF1LinearScalingBinaryClassificationModel,
    PositiveF1LinearScalingBinaryClassificationModel,
    PositiveNegativeRatesLinearScalingBinaryClassificationModel,
    PredictiveValuesLinearScalingBinaryClassificationModel,
    RighteousnessLinearScalingBinaryClassificationModel,
)
from caliber.binary_classification.metrics import (
    average_smooth_squared_calibration_error,
    average_squared_calibration_error,
    expected_calibration_error,
)
from data import load_breast_cancer_data, load_heart_disease_data, load_two_moons_data

THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5
N_GROUPS = 5


datasets = {
    "heart_disease": load_heart_disease_data(),
    "two_moons": load_two_moons_data(),
    "breast_cancer": load_breast_cancer_data(),
}

for dataset_name, dataset in datasets.items():
    train_inputs, test_inputs, train_targets, test_targets = dataset

    clustering_model = GaussianMixture(n_components=N_GROUPS)
    clustering_model.fit(train_inputs)
    train_group_scores = clustering_model.predict_proba(train_inputs)
    test_group_scores = clustering_model.predict_proba(test_inputs)
    train_group_preds = clustering_model.predict(train_inputs)
    test_group_preds = clustering_model.predict(test_inputs)
    train_groups = np.zeros((len(train_group_preds), N_GROUPS)).astype(bool)
    test_groups = np.zeros((len(test_group_preds), N_GROUPS)).astype(bool)
    train_groups[np.arange(len(train_groups)), train_group_preds] = True
    test_groups[np.arange(len(test_groups)), test_group_preds] = True

    train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
    train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
    train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]
    train_groups, val_groups = train_groups[:train_size], train_groups[train_size:]

    model = XGBClassifier(random_state=42)
    model.fit(train_inputs, train_targets)

    val_probs = model.predict_proba(val_inputs)[:, 1]
    test_probs = model.predict_proba(test_inputs)[:, 1]
    test_preds = (test_probs >= THRESHOLD).astype(int)

    posthoc_models = {
        "balanced_accuracy_linear_scaling": BalancedAccuracyLinearScalingBinaryClassificationModel(
            threshold=THRESHOLD
        ),
        "balanced_accuracy_temperature_scaling": BalancedAccuracyLinearScalingBinaryClassificationModel(
            threshold=THRESHOLD, has_intercept=False
        ),
        "positive_f1_linear_scaling": PositiveF1LinearScalingBinaryClassificationModel(
            threshold=THRESHOLD
        ),
        "positive_f1_temperature_scaling": PositiveF1LinearScalingBinaryClassificationModel(
            threshold=THRESHOLD, has_intercept=False
        ),
        "negative_f1_linear_scaling": NegativeF1LinearScalingBinaryClassificationModel(
            threshold=THRESHOLD
        ),
        "negative_f1_temperature_scaling": NegativeF1LinearScalingBinaryClassificationModel(
            threshold=THRESHOLD, has_intercept=False
        ),
        "predictive_values_linear_scaling": PredictiveValuesLinearScalingBinaryClassificationModel(
            threshold=THRESHOLD
        ),
        "predictive_values_temperature_scaling": PredictiveValuesLinearScalingBinaryClassificationModel(
            threshold=THRESHOLD, has_intercept=False
        ),
        "positive_negative_rates_linear_scaling": PositiveNegativeRatesLinearScalingBinaryClassificationModel(
            threshold=THRESHOLD
        ),
        "positive_negative_rates_temperature_scaling": PositiveNegativeRatesLinearScalingBinaryClassificationModel(
            threshold=THRESHOLD, has_intercept=False
        ),
        "righteousness_linear_scaling": RighteousnessLinearScalingBinaryClassificationModel(
            threshold=THRESHOLD
        ),
        "righteousness_temperature_scaling": RighteousnessLinearScalingBinaryClassificationModel(
            threshold=THRESHOLD, has_intercept=False
        ),
        "brier_linear_scaling": BrierLinearScalingBinaryClassificationModel(),
        "brier_temperature_scaling": BrierLinearScalingBinaryClassificationModel(
            has_intercept=False
        ),
        "cross_entropy_linear_scaling": CrossEntropyLinearScalingBinaryClassificationModel(),
        "cross_entropy_temperature_scaling": CrossEntropyLinearScalingBinaryClassificationModel(
            has_intercept=False
        ),
        "focal_linear_scaling": FocalLinearScalingBinaryClassificationModel(),
        "focal_temperature_scaling": FocalLinearScalingBinaryClassificationModel(
            has_intercept=False
        ),
        "constant_shift": ModelBiasBinaryClassificationConstantShift(),
        "histogram_binning": HistogramBinningBinaryClassificationModel(),
        "isotonic_regression": IsotonicRegressionBinaryClassificationModel(),
        "iterative_smooth_histogram_binning": IterativeSmoothHistogramBinningBinaryClassificationModel(),
        "iterative_smooth_grouped_histogram_binning": IterativeSmoothHistogramBinningBinaryClassificationModel(),
        "iterative_histogram_binning": IterativeBinningBinaryClassificationModel(),
        "iterative_linear_binning": IterativeBinningBinaryClassificationModel(
            bin_model=BrierLinearScalingBinaryClassificationModel(),
        ),
        "iterative_grouped_linear_binning": IterativeBinningBinaryClassificationModel(
            bin_model=BrierLinearScalingBinaryClassificationModel(),
        ),
        "grouped_conditional_unbiased": GroupConditionalUnbiasedBinaryClassificationModel(),
        "iterative_grouped_fitting": IterativeFittingBinaryClassificationModel(),
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
        "ASSCE": average_smooth_squared_calibration_error,
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
        if "grouped" not in m_name:
            m.fit(val_probs, val_targets)
            posthoc_test_probs = m.predict_proba(test_probs)
            posthoc_test_preds = m.predict(test_probs)
        else:
            m.fit(val_probs, val_targets, val_groups)
            posthoc_test_probs = m.predict_proba(test_probs, test_groups)
            posthoc_test_preds = m.predict(test_probs, test_groups)
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
