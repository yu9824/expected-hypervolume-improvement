from typing import Literal, Callable, Optional, Union, Any, overload
from typing import List, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from utils import create_vw


class MultiGaussianProcessRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_models: int = 10,
        kernel: Optional[Kernel] = None,
        *,
        alpha: Union[float, np.ndarray] = 1e-10,
        optimizer: Optional[
            Union[Literal["fmin_l_bfgs_b", "fmin_l_bfgs_b"], Callable]
        ] = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        copy_X_train: bool = True,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
    ) -> None:
        super().__init__()
        self.n_models = n_models
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.__models = []
        for _ in range(n_models):
            self.__models.append(
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    optimizer=optimizer,
                    n_restarts_optimizer=n_restarts_optimizer,
                    normalize_y=normalize_y,
                    copy_X_train=copy_X_train,
                    random_state=random_state,
                )
            )

    def fit(self, X: List[List[float]], y: List[List[float]]):
        X: np.ndarray = check_array(X, ensure_2d=True)
        y: np.ndarray = check_array(y, ensure_2d=True)
        assert y.shape[1] == self.n_models
        for _model, _y in zip(self.models, y.transpose()):
            _model.fit(X, _y)
        return self

    @overload
    def predict(
        self, X: List[List[float]], return_std: True
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @overload
    def predict(self, X: List[List[float]], return_std: False) -> np.ndarray:
        pass

    def predict(
        self,
        X: List[List[float]],
        return_std: bool = False,
    ) -> Any:
        mean = []
        if return_std:
            std = []
        for model in self.models:
            if return_std:
                _mean, _std = model.predict(X, return_std=return_std)
                std.append(_std)
            else:
                _mean = model.predict(X, return_std=return_std)
            mean.append(_mean)
        if return_std:
            return np.array(mean).transpose(), np.array(std).transpose()
        else:
            return np.array(mean).transpose()

    @property
    def models(self) -> List[GaussianProcessRegressor]:
        return self.__models


class ExpectedHypervolumeImprovement:
    def __init__(
        self,
        model: MultiGaussianProcessRegressor,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        self.model = model
        self.X_train: np.ndarray = check_array(X_train, ensure_2d=True)
        self.y_train: np.ndarray = check_array(y_train, ensure_2d=True)

        self.model.fit(self.X_train, self.y_train)

        self.w_ref = np.max(self.y_train, axis=0) + 1
        self.v_ref = np.min(self.y_train, axis=0) - 1
        self.n_tasks = self.y_train.shape[1]
        self.n_train = self.y_train.shape[0]
        # cellの作成
        self.v, self.w = create_vw(self.y_train, self.v_ref, self.w_ref)

    def __call__(self, X: np.ndarray) -> List[float]:
        mean, std = self.model.predict(X, return_std=True)
        mean = mean[:, np.newaxis, :]
        std = std[:, np.newaxis, :]
        alpha = (mean - self.v) / std
        beta = (mean - self.w) / std

        ehvi_each_cell = std**2 * (
            (norm.pdf(beta) - norm.pdf(alpha))
            + beta * (norm.cdf(beta) - norm.cdf(alpha))
        )
        ehvi = np.sum(np.prod(ehvi_each_cell, axis=2), axis=1)

        assert X.shape[0] == ehvi.shape[0]
        for i in range(ehvi.shape[0]):
            if np.any(np.all(self.X_train == X[i], axis=1)):
                ehvi[i] = float("inf")
        return -ehvi


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    data = load_diabetes()
    X = data.data
    y = data.target
    y2 = data.target / 10 + np.random.randn(len(y))
    X_train, X_test, y_train, y_test, y2_train, y2_test = train_test_split(
        X, y, y2, test_size=0.2, random_state=0
    )
    expected_hypervolume_improvement = ExpectedHypervolumeImprovement(
        MultiGaussianProcessRegressor(2),
        X_train,
        np.vstack([y_train, y2_train]).transpose(),
    )
    print(expected_hypervolume_improvement(X_test))
