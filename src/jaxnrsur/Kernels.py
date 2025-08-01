"""
Kernels module for Gaussian process and related models using JAX and Equinox.

This module defines base and composite kernel classes, as well as specific kernel implementations
such as constant, white noise, and RBF kernels. Kernels are used to compute covariance matrices
for Gaussian process models.
"""

from abc import abstractmethod
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class Kernel(eqx.Module):
    """
    Abstract base class for all kernels.

    Methods:
        __call__(X, Y): Compute the kernel between two sets of samples.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        """
        Compute the kernel between two sets of samples.

        Args:
            X (Float[Array, " n_sample"]): First set of samples.
            Y (Float[Array, " n_sample"]): Second set of samples.

        Returns:
            Float[Array, " n_sample"]: Kernel matrix.
        """


class SumKernel(Kernel):
    """
    Kernel representing the sum of two kernels.

    Attributes:
        k1 (Kernel): First kernel.
        k2 (Kernel): Second kernel.
    """

    k1: Kernel
    k2: Kernel

    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        """
        Compute the sum of two kernels.

        Args:
            X (Float[Array, " n_sample"]): First set of samples.
            Y (Float[Array, " n_sample"]): Second set of samples.

        Returns:
            Float[Array, " n_sample"]: Kernel matrix.
        """
        return self.k1(X, Y) + self.k2(X, Y)


class ProductKernel(Kernel):
    """
    Kernel representing the product of two kernels.

    Attributes:
        k1 (Kernel): First kernel.
        k2 (Kernel): Second kernel.
    """

    k1: Kernel
    k2: Kernel

    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        """
        Compute the product of two kernels.

        Args:
            X (Float[Array, " n_sample"]): First set of samples.
            Y (Float[Array, " n_sample"]): Second set of samples.

        Returns:
            Float[Array, " n_sample"]: Kernel matrix.
        """
        return self.k1(X, Y) * self.k2(X, Y)


class ConstantKernel(Kernel):
    """
    Kernel that returns a constant value for all entries.

    Attributes:
        constant_value (float): The constant value to return.
        x_dims (int): Number of rows in the output matrix.
        y_dims (int): Number of columns in the output matrix.
    """

    constant_value: float
    x_dims: int
    y_dims: int

    def __init__(self, constant_value: float = 1.0, x_dims: int = 1, y_dims: int = 1):
        super().__init__()
        self.constant_value = constant_value
        self.x_dims = x_dims
        self.y_dims = y_dims

    def with_params(self, params: dict):
        """
        Return a new ConstantKernel with updated parameters from a dictionary.

        Args:
            params (dict): Dictionary containing kernel parameters.

        Returns:
            ConstantKernel: New kernel instance with updated parameters.
        """
        if "name" in params:
            if "ConstantKernel" not in params["name"]:
                raise ValueError("Wrong Kernel name")
        else:
            raise KeyError("No name given")
        return ConstantKernel(
            constant_value=params.get("constant_value", self.constant_value),
            x_dims=params.get("x_dims", self.x_dims),
            y_dims=params.get("y_dims", self.y_dims),
        )

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        """
        Return a constant matrix of shape (x_dims, y_dims) filled with constant_value.

        Args:
            X (Float[Array, " n_sample"]): First set of samples (unused).
            Y (Float[Array, " n_sample"]): Second set of samples (unused).

        Returns:
            Float[Array, " n_sample"]: Constant matrix.
        """
        return jnp.full((self.x_dims, self.y_dims), self.constant_value)


class WhiteKernel(Kernel):
    """
    Kernel that returns a matrix of zeros, representing white noise.

    Attributes:
        noise_level (float): Noise level (currently unused).
        x_dims (int): Number of rows in the output matrix.
        y_dims (int): Number of columns in the output matrix.
    """

    noise_level: float
    x_dims: int
    y_dims: int

    def __init__(self, noise_level=1.0, x_dims: int = 1, y_dims: int = 1):
        super().__init__()
        self.noise_level = noise_level
        self.x_dims = x_dims
        self.y_dims = y_dims

    def with_params(self, params: dict):
        """
        Return a new WhiteKernel with updated parameters from a dictionary.

        Args:
            params (dict): Dictionary containing kernel parameters.

        Returns:
            WhiteKernel: New kernel instance with updated parameters.
        """
        if "name" in params:
            if "WhiteKernel" not in params["name"]:
                raise ValueError("Wrong Kernel name")
        else:
            raise KeyError("No name given")
        return WhiteKernel(
            noise_level=params.get("noise_level", self.noise_level),
            x_dims=params.get("x_dims", self.x_dims),
            y_dims=params.get("y_dims", self.y_dims),
        )

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        """
        Return a matrix of zeros of shape (x_dims, y_dims).

        Args:
            X (Float[Array, " n_sample"]): First set of samples (unused).
            Y (Float[Array, " n_sample"]): Second set of samples (unused).

        Returns:
            Float[Array, " n_sample"]: Zero matrix.
        """
        return jnp.zeros((self.x_dims, self.y_dims))


def cdist(
    X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
) -> Float[Array, " n_sample"]:
    """
    Compute squared Euclidean distance between two sets of samples.

    Args:
        X (Float[Array, " n_sample"]): First set of samples.
        Y (Float[Array, " n_sample"]): Second set of samples.

    Returns:
        Float[Array, " n_sample"]: Matrix of squared distances.
    """
    return jnp.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)


class RBF(Kernel):
    """
    Radial Basis Function (RBF) kernel (a.k.a. Gaussian kernel).

    Attributes:
        length_scale (float): Length scale parameter for the kernel.
    """

    length_scale: float

    def __init__(self, length_scale: float = 1.0):
        """
        Initialize the RBF kernel.

        Args:
            length_scale (float): Length scale parameter.
        """
        super().__init__()
        self.length_scale = length_scale

    def with_params(self, params: dict):
        """
        Return a new RBF kernel with updated parameters from a dictionary.

        Args:
            params (dict): Dictionary containing kernel parameters.

        Returns:
            RBF: New kernel instance with updated parameters.
        """
        if "name" in params:
            if "RBF" not in params["name"]:
                raise ValueError("Wrong Kernel name")
        else:
            raise KeyError("No name given")
        return RBF(
            length_scale=params.get("length_scale", self.length_scale),
        )

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        """
        Compute the RBF kernel between two sets of samples.

        Args:
            X (Float[Array, " n_sample"]): First set of samples.
            Y (Float[Array, " n_sample"]): Second set of samples.

        Returns:
            Float[Array, " n_sample"]: Kernel matrix.
        """
        dists = cdist((X / self.length_scale), (Y / self.length_scale))
        return jnp.exp(-0.5 * dists)
