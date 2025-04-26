from abc import abstractmethod
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class Kernel(eqx.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        """
        We currently support case when both arguments are supplied
        """


class SumKernel(Kernel):
    k1: Kernel
    k2: Kernel

    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        return self.k1(X, Y) + self.k2(X, Y)


class ProductKernel(Kernel):
    k1: Kernel
    k2: Kernel

    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        return self.k1(X, Y) * self.k2(X, Y)


class ConstantKernel(Kernel):
    constant_value: float
    x_dims: int
    y_dims: int

    def __init__(self, constant_value: float = 1.0, x_dims: int = 1, y_dims: int = 1):
        super().__init__()
        self.constant_value = constant_value
        self.x_dims = x_dims
        self.y_dims = y_dims

    def load_params(self, params: dict):
        keys = params.keys()
        if "name" in params:
            if "ConstantKernel" not in params["name"]:
                raise ValueError("Wrong Kernel name")
        else:
            raise KeyError("No name given")
        if "constant_value" in keys:
            self.constant_value = params["constant_value"]
            print("update constant value")
        if "x_dims" in keys:
            self.x_dims = params["x_dims"]
            print("update x_dims")
        if "y_dims" in keys:
            self.y_dims = params["y_dims"]
            print("update y_dims")

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        return jnp.full((self.x_dims, self.y_dims), self.constant_value)


class WhiteKernel(Kernel):
    noise_level: float
    x_dims: int
    y_dims: int

    def __init__(self, noise_level=1.0, x_dims: int = 1, y_dims: int = 1):
        super().__init__()
        self.noise_level = noise_level
        self.x_dims = x_dims
        self.y_dims = y_dims

    def load_params(self, params: dict):
        keys = params.keys()
        if "name" in params:
            if "WhiteKernel" not in params["name"]:
                raise ValueError("Wrong Kernel name")
        else:
            raise KeyError("No name given")
        if "noise_level" in keys:
            self.noise_level = params["noise_level"]
            print("update noise_level")
        if "x_dims" in keys:
            self.x_dims = params["x_dims"]
            print("update x_dims")
        if "y_dims" in keys:
            self.y_dims = params["y_dims"]
            print("update y_dims")

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        return jnp.zeros((self.x_dims, self.y_dims))


def cdist(
    X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
) -> Float[Array, " n_sample"]:
    """
    Compute square distance between two metric
    """
    return jnp.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)


class RBF(Kernel):
    length_scale: float

    """
    Anisotropic RBF kernel
    """

    def __init__(self, length_scale: float = 1.0):
        super().__init__()
        self.length_scale = length_scale

    def load_params(self, params: dict):
        keys = params.keys()
        if "name" in params:
            if "RBF" not in params["name"]:
                raise ValueError("Wrong Kernel name")
        else:
            raise KeyError("No name given")
        if "length_scale" in keys:
            self.length_scale = params["length_scale"]
            print("update length_scale")

    def __call__(
        self, X: Float[Array, " n_sample"], Y: Float[Array, " n_sample"]
    ) -> Float[Array, " n_sample"]:
        dists = cdist((X / self.length_scale), (Y / self.length_scale))
        return jnp.exp(-0.5 * dists)
