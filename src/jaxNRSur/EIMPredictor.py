import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxNRSur import Kernels
import equinox as eqx


class GaussianProcessRegressor(eqx.Module):
    kernel: Kernels.Kernel
    x_train: Float[Array, " n_samples"]
    y_train_mean: float
    y_train_std: float
    alpha: Float[Array, " n_samples"]
    L: Float[Array, " n_samples"]

    def __init__(self, parameters: dict):
        super().__init__()
        kernel_params = parameters["DICT_kernel_"]
        self.x_train = parameters["X_train_"]
        self.y_train_mean = parameters["_y_train_mean"]
        self.y_train_std = parameters["_y_train_std"]
        self.alpha = parameters["alpha_"]
        self.L = parameters[
            "L_"
        ]  # I think L is only needed when you want to predict the variance as well.

        self.kernel = self.compose_kernel(kernel_params)

    def predict_mean(self, params: Float[Array, " n_features"]):
        K_trans = self.kernel(params, self.x_train)
        y_mean = K_trans @ self.alpha
        y_mean = self.y_train_std * y_mean + self.y_train_mean
        return y_mean

    def compose_kernel(self, params: dict) -> Kernels.Kernel:
        if params["name"] == b"Sum":
            return Kernels.SumKernel(
                self.compose_kernel(params["DICT_k1"]),
                self.compose_kernel(params["DICT_k2"]),
            )
        elif params["name"] == b"Product":
            return Kernels.ProductKernel(
                self.compose_kernel(params["DICT_k1"]),
                self.compose_kernel(params["DICT_k2"]),
            )
        elif params["name"] == b"ConstantKernel":
            return Kernels.ConstantKernel(
                params["constant_value"], 1, self.x_train.shape[0]
            )
        elif params["name"] == b"WhiteKernel":
            return Kernels.WhiteKernel(params["noise_level"], 1, self.x_train.shape[0])
        elif params["name"] == b"RBF":
            return Kernels.RBF(params["length_scale"])
        else:
            raise NotImplementedError("Kernel not registered")


class LinearRegressor(eqx.Module):
    coefficient: Float[Array, " n_features"]
    intercept: float

    def __init__(self, parameters: dict):
        super().__init__()
        self.coefficient = jnp.array(parameters["coef_"])
        self.intercept = parameters["intercept_"]

    def __call__(self, params: Float[Array, " n_features"]):
        return (self.coefficient * params).sum() + self.intercept


class EIMpredictor(eqx.Module):
    data_mean: float
    data_std: float
    GPR_obj: GaussianProcessRegressor
    linearModel: LinearRegressor | None

    def __init__(self, data: dict, **kwargs):
        super().__init__(**kwargs)
        GPR_param = data["DICT_GPR_params"]
        self.data_mean = data["data_mean"]
        self.data_std = data["data_std"]

        if "_y_train_std" not in GPR_param:
            GPR_param["_y_train_std"] = 1

        # load GPR fit
        self.GPR_obj = GaussianProcessRegressor(GPR_param)

        # load LinearRegression fit
        lin_reg_params = data["DICT_lin_reg_params"]
        if lin_reg_params is not None:
            self.linearModel = LinearRegressor(lin_reg_params)
        else:
            self.linearModel = None

    def undo_normalization(
        self,
        data_normed: Float[Array, " n_samples"],
        data_normed_err: Float[Array, " n_samples"] | None = None,
    ) -> tuple[Float[Array, " n_samples"], Float[Array, " n_samples"] | None]:
        mean = data_normed * self.data_std + self.data_mean
        error = data_normed_err * self.data_std if data_normed_err is not None else None
        return mean, error

    def predict(self, X: Float[Array, " n_samples"]) -> Float[Array, " n_samples"]:
        y_normed_pred = self.GPR_obj.predict_mean(X)
        y_pred = self.undo_normalization(y_normed_pred)[0]
        y_pred = (
            y_pred + self.linearModel(X) if self.linearModel is not None else y_pred
        )
        return y_pred

    def __call__(self, X: Float[Array, " n_samples"]) -> Float[Array, " n_samples"]:
        return self.predict(X)
