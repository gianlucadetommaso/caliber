import numpy as np
import pytest
from numpy.typing import NDArray

from caliber import MahalanobisBinaryClassificationModel, PropensityScoreEstimationModel
from data import load_two_moons_data

METHODS = {
    "mahalanobis_without_targets": MahalanobisBinaryClassificationModel(threshold=0.5),
    "mahalanobis_with_targets": MahalanobisBinaryClassificationModel(threshold=0.5),
}

PROPENSITY_SCORE_METHODS = {"psem": PropensityScoreEstimationModel()}

train_inputs, test_inputs, train_targets, test_targets = load_two_moons_data()


@pytest.mark.parametrize("method_name", list(METHODS.keys()))
def test_method(method_name: str) -> None:
    m = METHODS[method_name]
    if method_name == "mahalanobis_with_targets":
        m.fit(train_inputs, train_targets)
    else:
        m.fit(train_inputs)
    test_probs = m.predict_proba(test_inputs)
    test_preds = m.predict(test_inputs)
    check_probs_preds(test_probs, test_preds)


@pytest.mark.parametrize("method_name", list(PROPENSITY_SCORE_METHODS.keys()))
def test_propensity_score_method(method_name: str) -> None:
    m = PROPENSITY_SCORE_METHODS[method_name]
    m.fit(train_inputs, train_targets)
    test_preds = m.predict(test_inputs)
    check_propensity_score_preds(test_preds)


def check_probs_preds(probs: NDArray[np.float64], preds: NDArray[np.float64]) -> None:
    assert probs.ndim == 1
    assert np.all(probs <= 1) and np.all(probs >= 0)
    assert preds.ndim == 1
    assert set(preds) in [{0, 1}, {0}, {1}]


def check_propensity_score_preds(preds: NDArray[np.float64]) -> None:
    assert preds.ndim == 1
