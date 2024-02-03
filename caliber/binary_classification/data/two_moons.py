from sklearn.datasets import make_moons


def load_two_moons_data(
    n_train_samples=1000, n_test_samples=1000, noise=0.1, random_state=0
):
    _train_inputs, _train_targets = make_moons(
        n_samples=n_train_samples, noise=noise, random_state=random_state
    )
    _test_inputs, _test_targets = make_moons(
        n_samples=n_test_samples, noise=noise, random_state=random_state + 1
    )
    return _train_inputs, _test_inputs, _train_targets, _test_targets
