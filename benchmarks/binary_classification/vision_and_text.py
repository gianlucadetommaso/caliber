import numpy as np
import torch
import os
import json
from tabulate import tabulate
from typing import Optional
from caliber import (
    BrierLinearScalingBinaryClassificationModel,
    BetaBinaryClassificationModel,
    FocalLinearScalingBinaryClassificationModel,
    HistogramBinningBinaryClassificationModel,
    IsotonicRegressionBinaryClassificationModel,
    IterativeBinningBinaryClassificationModel,
    IterativeKernelizedBinningBinaryClassificationModel,
    IterativeFittingBinaryClassificationModel,
    GroupConditionalUnbiasedBinaryClassificationModel,
    SmoothLinearScalingBinaryClassificationModel,
    OneShotKernelizedBinaryClassificationModel
)
from sklearn.mixture import GaussianMixture
from caliber.binary_classification.metrics import (
    average_smooth_squared_calibration_error,
    average_squared_calibration_error,
    expected_calibration_error,
    grouped_average_squared_calibration_error
)
from tqdm import tqdm


CALIB_FRAC = 0.5
DATA_DIR = "/Users/gianluca.detommaso/predictions/"
METRICS_DIR = "/Users/gianluca.detommaso/caliber/benchmarks/binary_classification"
DO_TRAIN = True
NUM_SEEDS = 30
WITH_GROUPS = True
GROUP_MODEL = GaussianMixture(n_components=5)
GROUP_SCORES_THRESHOLD = 0.8
METRICS_TO_PRINT = ["gasce"]

MODELS = {
    "uncalib": None,
    "beta": BetaBinaryClassificationModel(),
    "hb": HistogramBinningBinaryClassificationModel(),
    #"ir": IsotonicRegressionBinaryClassificationModel(),
    #"focal": FocalLinearScalingBinaryClassificationModel(),
    "sl": SmoothLinearScalingBinaryClassificationModel(),
    #"gcu": GroupConditionalUnbiasedBinaryClassificationModel(),
    #"ihb": IterativeBinningBinaryClassificationModel(),
    "ibls": IterativeBinningBinaryClassificationModel(
        bin_model=BrierLinearScalingBinaryClassificationModel(),
    ),
    #"if": IterativeFittingBinaryClassificationModel(),
    "osk": OneShotKernelizedBinaryClassificationModel(),
    "ik": IterativeKernelizedBinningBinaryClassificationModel(),
}

metrics_filename = "metrics" + ("_w_" if WITH_GROUPS else "_wo_") + "groups.json" 


def _init_metrics() -> dict[str, list[float]]:
    _metrics = dict(asce=[], assce=[], ece=[])
    if WITH_GROUPS:
        _metrics["gasce"] = []
    return _metrics


def _update_metrics(_metrics: dict[str, list[float]], targets: np.ndarray, probs: np.ndarray, groups: Optional[np.ndarray] = None) -> dict[str, list[float]]:
    _metrics["asce"].append(average_squared_calibration_error(targets, probs))
    _metrics["assce"].append(average_smooth_squared_calibration_error(targets, probs))
    _metrics["ece"].append(expected_calibration_error(targets, probs))
    if WITH_GROUPS:
        _metrics["gasce"].append(grouped_average_squared_calibration_error(targets, probs, groups))
    return _metrics


def _avg_metrics(_metrics: dict[str, list[float]]) -> dict[str, float]:
    for key, vals in _metrics.items():
        if key != "gasce":
            _metrics[key] = np.mean(vals)
        else:
            _metrics[key] = np.mean(vals, 0).tolist()
    return _metrics


if DO_TRAIN:
    metrics = dict(text=dict(), vision=dict())

    for dir_name in tqdm(os.listdir(DATA_DIR), desc="Dataset"):
        data_dir = os.path.join(DATA_DIR, dir_name)
        if not os.path.isdir(data_dir):
            continue
        filename = os.path.join(data_dir, os.listdir(data_dir)[0])
        data = torch.load(filename)

        if "Target" not in data:
            continue
        targets = data["Target"]
        if "AuxiliaryPredictionProb" not in data:
            continue

        print(f"Dataset: {dir_name}")
        dataset_type = "text" if "Text" in data else "vision"
        metrics[dataset_type][dir_name] = dict()

        probs = data["AuxiliaryPredictionProb"][:, 0]
        indices = np.argmax(probs, axis=1)
        probs = np.max(probs, axis=1)
        targets = np.array(targets == indices, dtype=int)
        features = data["Features"]
        
        
        for model_name, model in tqdm(MODELS.items(), desc="Model"):
            if model_name not in metrics[dataset_type][dir_name]:
                metrics[dataset_type][dir_name][model_name] = _init_metrics()
                
                for seed in tqdm(range(NUM_SEEDS), desc="Seed"):
                    rng = np.random.default_rng(seed)
                    perm = rng.choice(len(targets), len(targets))
                    targets = targets[perm]
                    probs = probs[perm]

                    calib_size = int(np.ceil(len(targets) * CALIB_FRAC))
                    calib_targets, test_targets = targets[:calib_size], targets[calib_size:]
                    calib_probs, test_probs = probs[:calib_size], probs[calib_size:]
                    calib_features, test_features = features[:calib_size], features[calib_size:]
                    
                    if WITH_GROUPS:
                        GROUP_MODEL.fit(calib_features)
                        calib_group_scores = GROUP_MODEL.predict_proba(calib_features)
                        group_threshold = np.quantile(calib_group_scores, GROUP_SCORES_THRESHOLD)
                        calib_groups = calib_group_scores > group_threshold
                        calib_groups = np.concatenate((calib_groups, np.ones((len(calib_groups), 1), dtype=bool)), axis=1)
                        test_group_scores = GROUP_MODEL.predict_proba(test_features)
                        test_group_binaries = test_group_scores > group_threshold
                        test_group_binaries = np.concatenate((test_group_binaries, np.ones((len(test_group_binaries), 1), dtype=bool)), axis=1)
                
                        if model_name == "uncalib":
                            new_test_probs = test_probs
                        elif model_name in ["beta", "hb", "ir", "focal", "sl"]:
                            model.fit(calib_probs, calib_targets)
                            new_test_probs = model.predict_proba(test_probs)
                        elif model_name in ["ihb", "ibls"]:
                            model.fit(calib_probs, calib_targets, calib_groups)
                            new_test_probs = model.predict_proba(test_probs, test_group_binaries)
                        elif model_name in ["gcu", "if", "osk", "ik"]:
                            if model_name == "gcu" and len(set(calib_targets)) == 1:
                                new_test_probs == 1
                            else:
                                model.fit(calib_probs, calib_targets, calib_group_scores)
                                new_test_probs = model.predict_proba(test_probs, test_group_scores)
                        else:
                            raise ValueError(f"model_name={model_name} not supported.")
                    else:
                        if model_name in ["uncalib", "gcu"]:
                            new_test_probs = test_probs
                        else:
                            model.fit(calib_probs, calib_targets)
                            new_test_probs = model.predict_proba(test_probs)
                    
                    metrics[dataset_type][dir_name][model_name] = _update_metrics(
                        metrics[dataset_type][dir_name][model_name], 
                        test_targets, 
                        new_test_probs,
                        groups=test_group_scores if WITH_GROUPS else None
                    )
                    
            metrics[dataset_type][dir_name][model_name] = _avg_metrics(metrics[dataset_type][dir_name][model_name])

        with open(os.path.join(METRICS_DIR, metrics_filename), "w") as json_file:
            json.dump(metrics, json_file)

metrics = json.load(open(os.path.join(METRICS_DIR, metrics_filename)))

for metric_name in METRICS_TO_PRINT:
    if metric_name != "gasce":
        for dataset_type, _metrics in sorted(metrics.items()):
            print(f"\n\n### Dataset type: {dataset_type}")
            dataset_names = list(_metrics.keys())
            model_names = list(_metrics[dataset_names[0]].keys())
            table = [[dataset_name] + [_metrics[dataset_name][model_name][metric_name] for model_name in model_names if model_name] for dataset_name in dataset_names]
            print(tabulate(table, tablefmt="rounded_outline", headers=[metric_name.upper()] + model_names))
    else:
        for _metrics in metrics.values():
            for dataset_name, dataset_metrics in sorted(_metrics.items()):
                print(f"Dataset: {dataset_name}")
                model_names = list(dataset_metrics.keys())
                table = [[f"Group: {i + 1}"] + [dataset_metrics[model_name][metric_name][i] for model_name in model_names] for i in range(len(dataset_metrics[model_names[0]][metric_name]))]
                print(tabulate(table, tablefmt="rounded_outline", headers=model_names))