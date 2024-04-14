from functools import partial

import numpy as np
from sklearn.linear_model import QuantileRegressor
from tabulate import tabulate

from caliber import ConformalizedQuantileRegressionModel
from caliber.regression.metrics import (
    prediction_interval_average_length,
    prediction_interval_coverage_probability,
)
from data import load_diabetes_data, load_regression_data

CONFIDENCE = 0.95
TRAIN_VAL_SPLIT = 0.5
MODEL_CLS = QuantileRegressor


datasets = {"diabetes": load_diabetes_data(), "regression": load_regression_data()}

for dataset_name, dataset in datasets.items():
    train_inputs, test_inputs, train_targets, test_targets = dataset

    train_size = int(len(train_inputs) * TRAIN_VAL_SPLIT)
    train_inputs, val_inputs = train_inputs[:train_size], train_inputs[train_size:]
    train_targets, val_targets = train_targets[:train_size], train_targets[train_size:]

    confidences = [1 - CONFIDENCE, CONFIDENCE]
    val_quantiles, test_quantiles = [], []
    for confidence in confidences:
        model = MODEL_CLS(quantile=confidence)
        model.fit(train_inputs, train_targets)
        val_quantiles.append(model.predict(val_inputs))
        test_quantiles.append(model.predict(test_inputs))
    val_quantiles = np.stack(val_quantiles, axis=1)
    test_quantiles = np.stack(test_quantiles, axis=1)

    calib_models = {
        "cqr": ConformalizedQuantileRegressionModel(
            confidence=CONFIDENCE,
        ),
    }
    calibration_metrics = {
        "picp": partial(
            prediction_interval_coverage_probability, interval_type="two-tailed"
        ),
        "pial": prediction_interval_average_length,
    }

    results = {
        **{MODEL_CLS.__name__: dict()},
        **{m_name: dict() for m_name, m in calib_models.items()},
    }

    for metric_name, metric in calibration_metrics.items():
        if metric_name == "pial":
            results[MODEL_CLS.__name__][metric_name] = metric(test_quantiles)
        else:
            results[MODEL_CLS.__name__][metric_name] = metric(
                test_targets, test_quantiles
            )

    for m_name, m in calib_models.items():
        m.fit(val_quantiles, val_targets)
        calib_test_quantiles = m.predict_interval(test_quantiles)
        for metric_name, metric in calibration_metrics.items():
            if metric_name == "pial":
                results[m_name][metric_name] = metric(calib_test_quantiles)
            else:
                results[m_name][metric_name] = metric(
                    test_targets, calib_test_quantiles
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
