import numpy as np
import torch

from caliber import (
    HistogramBinningBinaryClassificationModel,
    PPIHistogramBinningBinaryClassificationModel,
)
from caliber.binary_classification.metrics import (
    average_squared_calibration_error,
    expected_calibration_error,
)

CALIB_FRAC = 0.5
LABELED_FRAC = 0.01

filename = (
    "/Users/gianluca.detommaso/caliber/benchmarks/ppi/allmodels___ViT-B-32_zero_shot.pt"
)
data = torch.load(filename)

data.keys()

targets = data["Target"]
probs = data["AuxiliaryPredictionProb"][:, 1]
indices = np.argmax(probs, axis=1)
probs = np.max(probs, axis=1)
targets = np.array(targets == indices, dtype=int)
pseudo_targets = data["AuxiliaryPredictionProb"][:, 0][np.arange(len(indices)), indices]

asces, eces, ppi_asces, ppi_eces = [], [], [], []
for seed in range(1000):
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

    labeled_size = int(np.ceil(len(calib_targets) * LABELED_FRAC))
    labeled_calib_indices = np.arange(labeled_size)
    labeled_calib_targets = calib_targets[:labeled_size]
    labeled_calib_probs = calib_probs[:labeled_size]

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
    ppi_asce = average_squared_calibration_error(test_targets, ppi_predicted_test_probs)
    ppi_ece = expected_calibration_error(test_targets, ppi_predicted_test_probs)

    eces.append(ece)
    asces.append(asce)
    ppi_eces.append(ppi_ece)
    ppi_asces.append(ppi_asce)

print(f"ASCE: {np.mean(asces)}, {np.std(asces)}")
print(f"PPI ASCE: {np.mean(ppi_asces)}, {np.std(ppi_asces)}")
print(f"ECE: {np.mean(eces)}, {np.std(eces)}")
print(f"PPI ECE: {np.mean(ppi_eces)}, {np.std(ppi_eces)}")
