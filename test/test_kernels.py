import pytest
import jax.numpy as jnp
import dataclasses
from jax import random
from jaxnrsur.Kernels import (
    Kernel,
    SumKernel,
    ProductKernel,
    ConstantKernel,
    WhiteKernel,
    RBF,
    cdist,
)

class TestConstantKernel:
    @pytest.fixture
    def kernel(self):
        return ConstantKernel(constant_value=3.14, x_dims=2, y_dims=3)

    @pytest.fixture
    def kernel_default(self):
        return ConstantKernel()

    def test_basic(self, kernel):
        X = jnp.zeros((2, 1))
        Y = jnp.zeros((3, 1))
        result = kernel(X, Y)
        assert result.shape == (2, 3)
        assert jnp.allclose(result, 3.14)

    def test_load_params_frozen(self, kernel_default):
        params = {"name": "ConstantKernel", "constant_value": 2.5, "x_dims": 4, "y_dims": 5}
        with pytest.raises(dataclasses.FrozenInstanceError):
            kernel_default.load_params(params)

    def test_load_params_wrong_name(self, kernel_default):
        params = {"name": "CompletelyWrong"}
        with pytest.raises(ValueError):
            kernel_default.load_params(params)

    def test_load_params_no_name(self, kernel_default):
        params = {"constant_value": 1.0}
        with pytest.raises(KeyError):
            kernel_default.load_params(params)

class TestWhiteKernel:
    @pytest.fixture
    def kernel(self):
        return WhiteKernel(noise_level=0.5, x_dims=2, y_dims=4)

    @pytest.fixture
    def kernel_default(self):
        return WhiteKernel()

    def test_basic(self, kernel):
        X = jnp.zeros((2, 1))
        Y = jnp.zeros((4, 1))
        result = kernel(X, Y)
        assert result.shape == (2, 4)
        assert jnp.allclose(result, 0.0)

    def test_load_params_frozen(self, kernel_default):
        params = {"name": "WhiteKernel", "noise_level": 0.2, "x_dims": 3, "y_dims": 2}
        with pytest.raises(dataclasses.FrozenInstanceError):
            kernel_default.load_params(params)

    def test_load_params_wrong_name(self, kernel_default):
        params = {"name": "CompletelyWrong"}
        with pytest.raises(ValueError):
            kernel_default.load_params(params)

    def test_load_params_no_name(self, kernel_default):
        params = {"noise_level": 1.0}
        with pytest.raises(KeyError):
            kernel_default.load_params(params)

class TestRBFKernel:
    @pytest.fixture
    def kernel(self):
        return RBF(length_scale=2.0)

    @pytest.fixture
    def kernel_default(self):
        return RBF()

    def test_basic(self, kernel):
        X = jnp.array([[0.0], [1.0]])
        Y = jnp.array([[0.0], [2.0]])
        result = kernel(X, Y)
        dists = jnp.array([[0.0, 4.0], [1.0, 1.0]])
        dists = dists / (2.0 ** 2)
        expected = jnp.exp(-0.5 * dists)
        assert result.shape == (2, 2)
        assert jnp.allclose(result, expected)

    def test_load_params_frozen(self, kernel_default):
        params = {"name": "RBF", "length_scale": 3.5}
        with pytest.raises(dataclasses.FrozenInstanceError):
            kernel_default.load_params(params)

    def test_load_params_wrong_name_noop(self, kernel_default):
        params = {"name": "CompletelyWrong"}
        with pytest.raises(ValueError):
            kernel_default.load_params(params)

    def test_load_params_no_name(self, kernel_default):
        params = {"length_scale": 1.0}
        with pytest.raises(KeyError):
            kernel_default.load_params(params)

    def test_symmetry(self, kernel_default):
        k = RBF(length_scale=1.0)
        X = jnp.array([[0.0], [1.0], [2.0]])
        result = k(X, X)
        assert jnp.allclose(result, result.T)

    def test_diagonal(self, kernel_default):
        k = RBF(length_scale=1.0)
        X = jnp.array([[0.0], [1.0], [2.0]])
        result = k(X, X)
        diag = jnp.diag(result)
        assert jnp.allclose(diag, 1.0)

class TestCompositeKernels:
    def test_sum_kernel(self):
        k1 = ConstantKernel(constant_value=2.0, x_dims=2, y_dims=2)
        k2 = ConstantKernel(constant_value=3.0, x_dims=2, y_dims=2)
        sum_k = SumKernel(k1, k2)
        X = jnp.zeros((2, 1))
        Y = jnp.zeros((2, 1))
        result = sum_k(X, Y)
        assert result.shape == (2, 2)
        assert jnp.allclose(result, 5.0)

    def test_product_kernel(self):
        k1 = ConstantKernel(constant_value=2.0, x_dims=2, y_dims=2)
        k2 = ConstantKernel(constant_value=3.0, x_dims=2, y_dims=2)
        prod_k = ProductKernel(k1, k2)
        X = jnp.zeros((2, 1))
        Y = jnp.zeros((2, 1))
        result = prod_k(X, Y)
        assert result.shape == (2, 2)
        assert jnp.allclose(result, 6.0)

    def test_sum_and_product_kernel_with_rbf_and_white(self):
        k1 = RBF(length_scale=1.0)
        k2 = WhiteKernel(x_dims=3, y_dims=3)
        sum_k = SumKernel(k1, k2)
        prod_k = ProductKernel(k1, k2)
        X = jnp.linspace(0, 1, 3).reshape(-1, 1)
        Y = jnp.linspace(0, 1, 3).reshape(-1, 1)
        sum_result = sum_k(X, Y)
        prod_result = prod_k(X, Y)
        assert jnp.allclose(sum_result, k1(X, Y))
        assert jnp.allclose(prod_result, 0.0)

class TestUtils:
    def test_cdist(self):
        X = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        Y = jnp.array([[0.0, 0.0], [0.0, 1.0]])
        result = cdist(X, Y)
        expected = jnp.array([[0.0, 1.0], [1.0, 2.0]])
        assert result.shape == (2, 2)
        assert jnp.allclose(result, expected)

class TestKernelBase:
    def test_kernel_abstract_call(self):
        class DummyKernel(Kernel):
            def __call__(self, X, Y):
                return X + Y
        k = DummyKernel()
        X = jnp.array([1.0])
        Y = jnp.array([2.0])
        assert jnp.allclose(k(X, Y), 3.0)
