import pandas as pd
from sklearn.model_selection import train_test_split


def load_iris_data(random_state: int = 0):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None)
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    df["species"] = df["species"].map(
        dict(zip(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], [0, 1, 2]))
    )
    feature_columns = [c for c in df.columns if c != "species"]
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        df[feature_columns].to_numpy().astype(float),
        df["species"].values,
        random_state=random_state,
    )
    return train_inputs, test_inputs, train_targets, test_targets
