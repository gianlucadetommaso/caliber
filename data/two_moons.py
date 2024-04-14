import numpy as np
from sklearn.datasets import make_moons


def load_two_moons_data(
    n_train_samples: int = 1000,
    n_test_samples: int = 1000,
    noise: float = 0.1,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_inputs, train_targets = make_moons(
        n_samples=n_train_samples, noise=noise, random_state=random_state
    )
    test_inputs, test_targets = make_moons(
        n_samples=n_test_samples, noise=noise, random_state=random_state + 1
    )
    return train_inputs, test_inputs, train_targets, test_targets
