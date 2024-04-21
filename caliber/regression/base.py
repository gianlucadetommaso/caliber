import abc


class AbstractRegressionModel:
    def __init__(self):
        self._params = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass
