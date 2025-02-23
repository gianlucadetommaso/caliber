from numpy.typing import NDArray
from copy import deepcopy
import numpy as np
from caliber.regression.conformal_regression.base import AbstractRegressionModel


class JackknifePlusRegressionModel(AbstractRegressionModel):
    def __init__(self, model, coverage: float, loo_size: int = 100, seed: int = 0, loo_prediction: bool = False) -> None:
        self._model = model
        self._coverage = coverage
        self._loo_size = loo_size
        self._loo_prediction = loo_prediction
        self._rng = np.random.default_rng(seed)
        
    def fit(self, inputs: NDArray[np.float64], targets: NDArray[np.float64]) -> np.ndarray:
        indices = self._rng.choice(self._loo_size, size=self._loo_size, replace=False)
        
        self._models = []
        self._loo_errors: NDArray[np.float64] = np.zeros((self._loo_size,))
        for val_idx in indices:
            train_indices = [idx for idx in range(len(inputs)) if idx != val_idx]
            train_inputs, train_targets = inputs[train_indices], targets[train_indices]
            val_inputs, val_targets = inputs[None, val_idx], targets[None, val_idx]
            
            model = deepcopy(self._model)
            model.fit(train_inputs, train_targets)
            
            val_preds = model.predict(val_inputs)
            if val_preds.ndim == 2:
                if val_preds.shape[1] != 1:
                    raise ValueError("This method is supported only for scalar targets.")
                val_preds = val_preds.squeeze(1)
                
            self._loo_errors[val_idx] = np.abs(val_targets - val_preds)[0]
            
            self._models.append(model)
        
        if not self._loo_prediction:
            self._model.fit(inputs, targets)
             

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._loo_prediction:
            preds = np.zeros(len(inputs))
            for model in [self._models[0]]:
                preds_i = model.predict(inputs)
                if preds_i.ndim == 2:
                    if preds_i.shape[1] != 1:
                        raise ValueError("This method is supported only for scalar targets.")
                    preds_i = preds_i.squeeze(1)
                preds += preds_i
            preds /= self._loo_size
            return preds
        
        return self._model.predict(inputs)
        
        
    def predict_quantiles(self, inputs: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        loo_preds = np.zeros((self._loo_size, len(inputs)))
        for i, model in enumerate(self._models):
            preds = model.predict(inputs)
            if preds.ndim == 2:
                if preds.shape[1] != 1:
                    raise ValueError("This method is supported only for scalar targets.")
                preds = preds.squeeze(1)
            loo_preds[i] = preds
            
        left = loo_preds - self._loo_errors[:, None]
        right = loo_preds + self._loo_errors[:, None]        
            
        qleft = np.quantile(left, q=1 - self._coverage, axis=0)
        qright = np.quantile(right, q=self._coverage, axis=0)
        return np.array(list(zip(qleft, qright)))