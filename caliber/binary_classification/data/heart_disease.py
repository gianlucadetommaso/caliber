import pandas as pd
from sklearn.model_selection import train_test_split


def load_heart_disease_data(random_state: int = 0):
    heart_disease_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age",
        "sex",
        "chest_pain_type",
        "resting_blood_pressure",
        "serum_cholesterol",
        "fasting_blood_sugar",
        "resting_electrocardiographic_results",
        "max_heart_rate",
        "exercise_induced_angina",
        "st_depression_induced_by_exercise",
        "slope_of_the_peak_exercise_st_segment",
        "number_of_major_vessels",
        "thal",
        "target",
    ]
    heart_disease_df = pd.read_csv(heart_disease_url, names=column_names)
    heart_disease_df["target"] = (heart_disease_df["target"] >= 1).astype(int)
    heart_disease_df = pd.get_dummies(heart_disease_df)
    feature_columns = [c for c in heart_disease_df.columns if c != "target"]
    _train_inputs, _test_inputs, _train_targets, _test_targets = train_test_split(
        heart_disease_df[feature_columns].to_numpy().astype(float),
        heart_disease_df["target"].values,
        random_state=random_state,
    )
    return _train_inputs, _test_inputs, _train_targets, _test_targets
