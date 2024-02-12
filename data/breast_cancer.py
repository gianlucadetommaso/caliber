from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_breast_cancer_data(test_size=0.1, random_state=0):
    data = load_breast_cancer()
    inputs = data.data
    targets = data.target
    _train_inputs, _test_inputs, _train_targets, _test_targets = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )
    return _train_inputs, _test_inputs, _train_targets, _test_targets
