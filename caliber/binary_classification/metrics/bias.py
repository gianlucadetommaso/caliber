import numpy as np


def model_bias(targets: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean(targets - probs))


def absolute_model_bias(targets: np.ndarray, probs: np.ndarray) -> float:
    return np.abs(model_bias(targets, probs))


def squared_model_bias(targets: np.ndarray, probs: np.ndarray) -> float:
    return model_bias(targets, probs) ** 2
