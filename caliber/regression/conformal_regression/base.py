import abc


class ConformalRegressionModel:
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict_interval(self, *args, **kwargs):
        pass
