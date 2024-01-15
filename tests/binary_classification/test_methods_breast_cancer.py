import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import brier_score_loss
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from caliber import (
    ASCEBinaryClassificationLinearScaling,
    BalancedAccuracyBinaryClassificationLinearScaling,
    BrierBinaryClassificationLinearScaling,
    CrossEntropyBinaryClassificationLinearScaling,
    ECEBinaryClassificationLinearScaling,
    HistogramBinningBinaryClassificationModel,
    IsotonicRegressionBinaryClassificationModel,
    IterativeBinningBinaryClassificationModel,
    ModelBiasBinaryClassificationConstantShift,
    NegativeF1BinaryClassificationLinearScaling,
    PositiveF1BinaryClassificationLinearScaling,
    PositiveNegativeRatesBinaryClassificationLinearScaling,
    PredictiveValuesBinaryClassificationLinearScaling,
    RighteousnessBinaryClassificationLinearScaling,
    SmoothHistogramBinningBinaryClassificationModel,
)

THRESHOLD = 0.5
TRAIN_VAL_SPLIT = 0.5
N_GROUPS = 3

METHODS = {
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
    "constant_shift": ModelBiasBinaryClassificationConstantShift(),
    "histogram_binning": HistogramBinningBinaryClassificationModel(),
    "smooth_histogram_binning": SmoothHistogramBinningBinaryClassificationModel(),
    "isotonic_regression": IsotonicRegressionBinaryClassificationModel(),
    "iterative_histogram_binning": IterativeBinningBinaryClassificationModel(),
    "iterative_linear_binning": IterativeBinningBinaryClassificationModel(
        bin_model=BrierBinaryClassificationLinearScaling(),
        bin_loss_fn=brier_score_loss,
    ),
}

GROUPED_METHODS = {
    "iterative_grouped_linear_binning": IterativeBinningBinaryClassificationModel(
        bin_model=BrierBinaryClassificationLinearScaling(),
        bin_loss_fn=brier_score_loss,
    ),
    "smooth_grouped_histogram_binning": SmoothHistogramBinningBinaryClassificationModel(),
}


def load_breast_cancer_data(test_size=0.1, random_state=0):
    data = load_breast_cancer()
    inputs = data.data
    targets = data.target
    _train_inputs, _test_inputs, _train_targets, _test_targets = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )
    return _train_inputs, _test_inputs, _train_targets, _test_targets


train_inputs, test_inputs, train_targets, test_targets = load_breast_cancer_data()


clustering_model = GaussianMixture(n_components=N_GROUPS)
clustering_model.fit(train_inputs)
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

model = MLPClassifier(random_state=42)
model.fit(train_inputs, train_targets)

val_probs = model.predict_proba(val_inputs)[:, 1]
test_probs = model.predict_proba(test_inputs)[:, 1]
test_preds = (test_probs >= THRESHOLD).astype(int)


@pytest.mark.parametrize("m", list(METHODS.values()))
def test_method(m):
    m.fit(val_probs, val_targets)
    m.predict_proba(test_probs)
    m.predict(test_probs)


@pytest.mark.parametrize("m", list(GROUPED_METHODS.values()))
def test_grouped_method(m):
    m.fit(val_probs, val_targets, val_groups)
    m.predict_proba(test_probs, test_groups)
    m.predict(test_probs, test_groups)
