import numpy as np
from sklearn.datasets import make_regression


def load_regression_data(
    n_train_samples: int = 1000,
    n_test_samples: int = 1000,
    n_features: int = 2,
    n_targets: float = 1,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_inputs, train_targets = make_regression(
        n_samples=n_train_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=random_state,
    )
    test_inputs, test_targets = make_regression(
        n_samples=n_test_samples,
        n_features=n_features,
        n_targets=n_targets,
        random_state=random_state + 1,
    )
    return train_inputs, test_inputs, train_targets, test_targets
