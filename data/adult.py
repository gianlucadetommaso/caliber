import pandas as pd
from sklearn.model_selection import train_test_split


def load_adult_data(random_state: int = 0):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    adult_data = pd.read_csv(
        url, header=None, names=column_names, na_values=" ?", skipinitialspace=True
    )
    adult_data["income"] = adult_data["income"].map({"<=50K": 0, ">50K": 1})
    adult_data = pd.get_dummies(adult_data)
    feature_columns = [c for c in adult_data.columns if c != "income"]
    _train_inputs, _test_inputs, _train_targets, _test_targets = train_test_split(
        adult_data[feature_columns].to_numpy().astype(float),
        adult_data["income"].values,
        random_state=random_state,
    )
    return _train_inputs, _test_inputs, _train_targets, _test_targets
