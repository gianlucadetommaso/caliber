import json
import os

import numpy as np
import torch
from tabulate import tabulate

from caliber import (
    HistogramBinningBinaryClassificationModel,
    PPIHistogramBinningBinaryClassificationModel,
)
from caliber.binary_classification.metrics import (
    average_squared_calibration_error,
    expected_calibration_error,
)

CALIB_FRAC = 0.5
LABELED_SIZE = 100
DATA_DIR = "/Users/gianluca.detommaso/predictions/"
METRICS_DIR = "/Users/gianluca.detommaso/caliber/benchmarks/ppi/"
DO_STRATIFIED_SAMPLING = True
DO_TRAIN = True

if DO_TRAIN:
    metrics = dict(text=dict(), vision=dict())

    for dir_name in os.listdir(DATA_DIR):
        data_dir = os.path.join(DATA_DIR, dir_name)
        if not os.path.isdir(data_dir):
            continue
        filename = os.path.join(data_dir, os.listdir(data_dir)[0])
        data = torch.load(filename)
        #####
        if "Text" in data:
            continue
        ####

        if "Target" not in data:
            continue
        targets = data["Target"]
        if "AuxiliaryPredictionProb" not in data:
            continue

        print(dir_name)

        probs = data["AuxiliaryPredictionProb"][:, 1]
        indices = np.argmax(probs, axis=1)
        probs = np.max(probs, axis=1)
        targets = np.array(targets == indices, dtype=int)
        pseudo_targets = data["AuxiliaryPredictionProb"][:, 0][
            np.arange(len(indices)), indices
        ]

        asces, eces, ppi_asces, ppi_eces = [], [], [], []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            perm = rng.choice(len(targets), len(targets))
            targets = targets[perm]
            probs = probs[perm]
            pseudo_targets = pseudo_targets[perm]

            calib_size = int(np.ceil(len(targets) * CALIB_FRAC))
            calib_targets, test_targets = targets[:calib_size], targets[calib_size:]
            calib_probs, test_probs = probs[:calib_size], probs[calib_size:]
            calib_pseudo_targets, test_pseudo_targets = (
                pseudo_targets[:calib_size],
                pseudo_targets[calib_size:],
            )

            if DO_STRATIFIED_SAMPLING:
                bin_edges = np.linspace(0, 1, 10 + 1)
                bin_indices = np.digitize(calib_probs, bin_edges)
                bin_size = int(np.ceil(LABELED_SIZE / len(bin_edges)))
                labeled_calib_indices = []
                budgets = np.ceil(
                    np.array([np.sum(bin_indices == i) for i in range(1, 12)])
                    / len(calib_probs)
                    * LABELED_SIZE
                ).astype(int)
                for budget, b in zip(budgets, range(1, 12)):
                    if budget == 0:
                        continue
                    idx = np.where(bin_indices == b)[0]
                    labeled_calib_indices += np.random.choice(
                        idx, budget, replace=False
                    ).tolist()
            else:
                labeled_calib_indices = np.arange(LABELED_SIZE)

            labeled_calib_targets = calib_targets[labeled_calib_indices]
            labeled_calib_probs = calib_probs[labeled_calib_indices]

            model = HistogramBinningBinaryClassificationModel()
            model.fit(labeled_calib_probs, labeled_calib_targets)
            predicted_test_probs = model.predict_proba(test_probs)
            asce = average_squared_calibration_error(test_targets, predicted_test_probs)
            ece = expected_calibration_error(test_targets, predicted_test_probs)

            ppi_model = PPIHistogramBinningBinaryClassificationModel()
            ppi_model.fit(
                calib_probs, calib_targets, calib_pseudo_targets, labeled_calib_indices
            )
            ppi_predicted_test_probs = ppi_model.predict_proba(test_probs)
            ppi_asce = average_squared_calibration_error(
                test_targets, ppi_predicted_test_probs
            )
            ppi_ece = expected_calibration_error(test_targets, ppi_predicted_test_probs)

            eces.append(ece)
            asces.append(asce)
            ppi_eces.append(ppi_ece)
            ppi_asces.append(ppi_asce)

        dataset_type = "text" if "Text" in data else "vision"
        metrics[dataset_type][dir_name] = dict(
            mean_asce=np.mean(asces),
            std_asce=np.std(asces),
            mean_ece=np.mean(eces),
            std_ece=np.std(eces),
            mean_ppi_asce=np.mean(ppi_asces),
            std_ppi_asce=np.std(ppi_asces),
            mean_ppi_ece=np.mean(ppi_eces),
            std_ppi_ece=np.std(ppi_eces),
        )

    with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as json_file:
        json.dump(metrics, json_file)

metrics = json.load(open(os.path.join(METRICS_DIR, "metrics.json")))
metrics = {
    t: {name: {k: round(v, 4) for k, v in m.items()} for name, m in _metrics.items()}
    for t, _metrics in metrics.items()
}

headers = ["", "ASCE", "PPI ASCE", "ECE", "PPI_ECE"]
for _metrics in metrics.values():
    table = [
        [
            k,
            f"{m['mean_asce']} ({m['std_asce']})",
            f"{m['mean_ppi_asce']} ({m['std_asce']})",
            f"{m['mean_ece']} ({m['std_ece']})",
            f"{m['mean_ppi_ece']} ({m['std_ece']})",
        ]
        for k, m in _metrics.items()
    ]
    print(tabulate(table, tablefmt="rounded_outline", headers=headers))
