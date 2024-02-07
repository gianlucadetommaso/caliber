from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digits_data(n_classes: int = 10, random_state: int = 0):
    data = load_digits(n_class=n_classes)
    train_inputs, train_targets = data["data"] / 16.0, data["target"]
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        train_inputs, train_targets, random_state=random_state
    )
    return train_inputs, test_inputs, train_targets, test_targets
