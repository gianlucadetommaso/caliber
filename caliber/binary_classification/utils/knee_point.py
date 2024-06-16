from typing import Tuple

import numpy as np


def knee_point(
    probs: np.ndarray, targets: np.ndarray, n_thresholds: int = 100
) -> Tuple[float, float, float]:
    thresholds = np.linspace(0, 1, n_thresholds)
    preds = probs[:, None] >= thresholds[None]
    n_pos_targets = np.sum(targets)
    n_pos_preds = np.sum(preds, 0)
    joints = np.sum(targets[:, None] * preds, 0)
    recalls = np.where(n_pos_targets > 0, joints / n_pos_targets, 0.0)
    conds = np.where(n_pos_preds > 0)[0]
    precisions = np.zeros(n_thresholds)
    precisions[conds] = joints[conds] / n_pos_preds[conds]

    idx = np.argmax(precisions + recalls)
    return float(precisions[idx]), float(recalls[idx]), float(thresholds[idx])
