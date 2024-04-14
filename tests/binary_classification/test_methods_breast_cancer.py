import numpy as np
import pytest
from sklearn.metrics import brier_score_loss
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier

from caliber import (
    ASCELinearScalingBinaryClassificationModel,
    BalancedAccuracyLinearScalingBinaryClassificationModel,
    BrierLinearScalingBinaryClassificationModel,
    CrossEntropyLinearScalingBinaryClassificationModel,
    ECELinearScalingBinaryClassificationModel,
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
from data import load_breast_cancer_data

THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5
N_GROUPS = 3

METHODS = {
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
    "asce_linear_scaling": ASCELinearScalingBinaryClassificationModel(),
    "asce_temperature_scaling": ASCELinearScalingBinaryClassificationModel(
        has_intercept=False
    ),
    "ece_linear_scaling": ECELinearScalingBinaryClassificationModel(),
    "ece_temperature_scaling": ECELinearScalingBinaryClassificationModel(
        has_intercept=False
    ),
    "constant_shift": ModelBiasBinaryClassificationConstantShift(),
    "histogram_binning": HistogramBinningBinaryClassificationModel(),
    "iterative_smooth_histogram_binning": IterativeSmoothHistogramBinningBinaryClassificationModel(),
    "isotonic_regression": IsotonicRegressionBinaryClassificationModel(),
    "iterative_histogram_binning": IterativeBinningBinaryClassificationModel(),
    "iterative_linear_binning": IterativeBinningBinaryClassificationModel(
        bin_model=BrierLinearScalingBinaryClassificationModel(),
        bin_loss_fn=brier_score_loss,
    ),
}

GROUPED_METHODS = {
    "iterative_grouped_linear_binning": IterativeBinningBinaryClassificationModel(
        bin_model=BrierLinearScalingBinaryClassificationModel(),
        bin_loss_fn=brier_score_loss,
    ),
    "iterative_grouped_fitting": IterativeFittingBinaryClassificationModel(),
    "grouped_conditional_unbiased": GroupConditionalUnbiasedBinaryClassificationModel(),
}

GROUP_SCORED_METHODS = {
    "iterative_smooth_grouped_histogram_binning": IterativeSmoothHistogramBinningBinaryClassificationModel(),
}


train_inputs, test_inputs, train_targets, test_targets = load_breast_cancer_data()


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
train_group_scores, val_group_scores = (
    train_group_scores[:train_size],
    train_group_scores[train_size:],
)
train_groups, val_groups = train_groups[:train_size], train_groups[train_size:]

model = MLPClassifier(random_state=42)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)[:, 1]
test_probs = model.predict_proba(test_inputs)[:, 1]
test_preds = (test_probs >= THRESHOLD).astype(int)


@pytest.mark.parametrize("m", list(METHODS.values()))
def test_method(m):
    m.fit(val_probs, val_targets)
    probs = m.predict_proba(test_probs)
    preds = m.predict(test_probs)
    check_probs_preds(probs, preds)


@pytest.mark.parametrize("m", list(GROUPED_METHODS.values()))
def test_grouped_method(m):
    m.fit(val_probs, val_targets, val_groups)
    probs = m.predict_proba(test_probs, test_groups)
    preds = m.predict(test_probs, test_groups)
    check_probs_preds(probs, preds)


@pytest.mark.parametrize("m", list(GROUP_SCORED_METHODS.values()))
def test_group_scored_method(m) -> None:
    m.fit(val_probs, val_targets, val_group_scores)
    probs = m.predict_proba(test_probs, test_group_scores)
    preds = m.predict(test_probs, test_group_scores)
    check_probs_preds(probs, preds)


def check_probs_preds(probs: np.ndarray, preds: np.ndarray) -> None:
    assert probs.ndim == 1
    assert np.all(probs <= 1) and np.all(probs >= 0)
    assert preds.ndim == 1
    assert set(preds) in [{0, 1}, {0}, {1}]
