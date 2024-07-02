import abc
from typing import Callable, Optional

from caliber.regression.base import AbstractRegressionModel


class MinimizingRegressionModel(AbstractRegressionModel):
    def __init__(
        self,
        loss_fn: Callable,
        minimize_options: Optional[dict] = None,
    ):
        super().__init__()
        self._loss_fn = loss_fn
        self._params = None
        self._minimize_options = self._config_minimize_options(minimize_options)

    @abc.abstractmethod
    def _config_minimize_options(self, minimize_options: Optional[dict]) -> dict:
        pass
