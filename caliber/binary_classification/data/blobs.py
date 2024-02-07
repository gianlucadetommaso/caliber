from sklearn.datasets import make_blobs


def load_blobs_data(n_train_samples=1000, n_test_samples=1000, random_state=0):
    _train_inputs, _train_targets = make_blobs(
        n_samples=n_train_samples,
        cluster_std=0.1,
        centers=2,
        center_box=(-1, 1),
        random_state=random_state,
    )
    _test_inputs, _test_targets = make_blobs(
        n_samples=n_test_samples,
        cluster_std=0.1,
        centers=2,
        center_box=(-1, 1),
        random_state=random_state + 1,
    )
    return _train_inputs, _test_inputs, _train_targets, _test_targets
