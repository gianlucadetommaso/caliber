import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_breast_cancer_data(
    test_size: float = 0.1, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    inputs = data.data
    targets = data.target
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )
    return train_inputs, test_inputs, train_targets, test_targets
