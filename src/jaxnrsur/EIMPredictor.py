import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxnrsur import Kernels
import equinox as eqx
from typing import Optional


class GaussianProcessRegressor(eqx.Module):
    """Gaussian Process Regressor using custom kernels.

    Attributes:
        kernel (Kernels.Kernel): The kernel function used for GPR.
        x_train (Float[Array, " n_samples"]): Training input data.
        y_train_mean (float): Mean of the training targets.
        y_train_std (float): Standard deviation of the training targets.
        alpha (Float[Array, " n_samples"]): Precomputed weights for prediction.
        L (Float[Array, " n_samples"]): Cholesky factor of the kernel matrix.
    """

    kernel: Kernels.Kernel
    x_train: Float[Array, " n_samples"]
    y_train_mean: float
    y_train_std: float
    alpha: Float[Array, " n_samples"]
    L: Float[Array, " n_samples"]

    def __init__(self, parameters: dict):
        """Initializes the GaussianProcessRegressor.

        Args:
            parameters (dict): Dictionary containing kernel parameters, training data, and precomputed values.
        """
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
        """Predicts the mean of the target for given input features.

        Args:
            params (Float[Array, " n_features"]): Input features for prediction.

        Returns:
            Float[Array, " n_samples"]: Predicted mean values.
        """
        K_trans = self.kernel(params, self.x_train)
        y_mean = K_trans @ self.alpha
        y_mean = self.y_train_std * y_mean + self.y_train_mean
        return y_mean

    def compose_kernel(self, params: dict) -> Kernels.Kernel:
        """Recursively composes kernel functions from parameters.

        Args:
            params (dict): Dictionary specifying the kernel structure and parameters.

        Returns:
            Kernels.Kernel: Composed kernel object.
        """
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
    """Simple linear regression model.

    Attributes:
        coefficient (Float[Array, " n_features"]): Regression coefficients.
        intercept (float): Regression intercept.
    """

    coefficient: Float[Array, " n_features"]
    intercept: float

    def __init__(self, parameters: dict):
        """Initializes the LinearRegressor.

        Args:
            parameters (dict): Dictionary containing coefficients and intercept.
        """
        super().__init__()
        self.coefficient = jnp.array(parameters["coef_"])
        self.intercept = parameters["intercept_"]

    def __call__(self, params: Float[Array, " n_features"]):
        """Predicts the target value for given input features.

        Args:
            params (Float[Array, " n_features"]): Input features.

        Returns:
            float: Predicted value.
        """
        return (self.coefficient * params).sum() + self.intercept


class EIMpredictor(eqx.Module):
    """Combined EIM predictor using GPR and optional linear regression.

    Attributes:
        data_mean (float): Mean of the original data.
        data_std (float): Standard deviation of the original data.
        GPR_obj (GaussianProcessRegressor): GPR model object.
        linearModel (Optional[LinearRegressor]): Linear regression model object.
    """

    data_mean: float
    data_std: float
    GPR_obj: GaussianProcessRegressor
    linearModel: Optional[LinearRegressor]

    def __init__(self, data: dict, **kwargs):
        """Initializes the EIMpredictor.

        Args:
            data (dict): Dictionary containing GPR and linear regression parameters, and normalization stats.
            **kwargs: Additional keyword arguments for Equinox module.
        """
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
        """Reverts normalization on data and error.

        Args:
            data_normed (Float[Array, " n_samples"]): Normalized data.
            data_normed_err (Float[Array, " n_samples"] | None): Normalized error (optional).

        Returns:
            tuple: Tuple of (denormalized data, denormalized error or None).
        """
        mean = data_normed * self.data_std + self.data_mean
        error = data_normed_err * self.data_std if data_normed_err is not None else None
        return mean, error

    def predict(self, X: Float[Array, " n_samples"]) -> Float[Array, " n_samples"]:
        """Predicts the target values for input data.

        Args:
            X (Float[Array, " n_samples"]): Input data for prediction.

        Returns:
            Float[Array, " n_samples"]: Predicted values.
        """
        y_normed_pred = self.GPR_obj.predict_mean(X)
        y_pred = self.undo_normalization(y_normed_pred)[0]
        y_pred = (
            y_pred + self.linearModel(X) if self.linearModel is not None else y_pred
        )
        return y_pred

    def __call__(self, X: Float[Array, " n_samples"]) -> Float[Array, " n_samples"]:
        """Calls the predictor on input data.

        Args:
            X (Float[Array, " n_samples"]): Input data.

        Returns:
            Float[Array, " n_samples"]: Predicted values.
        """
        return self.predict(X)
