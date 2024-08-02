from functools import partial

import numpy as np
from sklearn.linear_model import QuantileRegressor
from tabulate import tabulate

from caliber import (
    ConformalizedQuantileRegressionModel,
    IterativeBinningMeanRegressionModel,
    IterativeBinningQuantileRegressionModel,
)
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

    confidences = [0.5 * (1 - CONFIDENCE), 0.5 * (1 + CONFIDENCE)]
    val_quantiles, test_quantiles = [], []
    for confidence in confidences:
        quantile_model = MODEL_CLS(quantile=confidence)
        quantile_model.fit(train_inputs, train_targets)
        val_quantiles.append(quantile_model.predict(val_inputs))
        test_quantiles.append(quantile_model.predict(test_inputs))
    val_quantiles = np.stack(val_quantiles, axis=1)
    test_quantiles = np.stack(test_quantiles, axis=1)

    pred_model = MODEL_CLS(quantile=0.5)
    pred_model.fit(train_inputs, train_targets)
    val_preds = pred_model.predict(val_inputs)
    test_preds = pred_model.predict(test_inputs)

    calib_models = {
        "cqr": ConformalizedQuantileRegressionModel(confidence=CONFIDENCE),
        "ibqr": IterativeBinningQuantileRegressionModel(
            confidence=CONFIDENCE, min_prob_bin=0
        ),
        "ibmr": IterativeBinningMeanRegressionModel(
            confidence=CONFIDENCE,
            min_prob_bin=0,
        ),
    }
    calibration_metrics = {
        "picp": partial(
            prediction_interval_coverage_probability, which_quantile="both"
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
        if m_name != "ibmr":
            m.fit(val_quantiles, val_targets)
            calib_test_quantiles = m.predict(test_quantiles)
            calib_val_quantiles = m.predict(val_quantiles)
        else:
            m.fit(val_preds, val_targets)
            calib_test_quantiles = m.predict(test_preds)
            calib_val_quantiles = m.predict(val_preds)
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
