import numpy as np
from sklearn.datasets import make_blobs


def load_blobs_data(
    n_train_samples: int = 1000, n_test_samples: int = 1000, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_inputs, train_targets = make_blobs(
        n_samples=n_train_samples,
        cluster_std=0.1,
        centers=2,
        center_box=(-1, 1),
        random_state=random_state,
    )
    test_inputs, test_targets = make_blobs(
        n_samples=n_test_samples,
        cluster_std=0.1,
        centers=2,
        center_box=(-1, 1),
        random_state=random_state + 1,
    )
    return train_inputs, test_inputs, train_targets, test_targets
