import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from jaxnrsur.Spline import CubicSpline


@pytest.fixture(scope="module")
def spline_fixture():
    x = jnp.linspace(0, 1, 5)
    y = x**2
    return CubicSpline(x, y)


def test_cubic_spline_basic(spline_fixture):
    # Check exact interpolation at grid points
    x_grid = jnp.linspace(0, 1, 5)
    y_grid = x_grid**2
    y_interp_grid = spline_fixture(x_grid)
    assert jnp.allclose(y_interp_grid, y_grid, atol=1e-8)

    # Check reasonable accuracy at off-grid points (looser tolerance)
    x_test = jnp.linspace(0, 1, 10)
    y_interp = spline_fixture(x_test)
    y_true = x_test**2
    max_err = jnp.max(jnp.abs(y_interp - y_true))
    print("Max error off-grid:", max_err)
    assert max_err < 0.03  # Allow a small error for natural cubic spline


def test_cubic_spline_grad(spline_fixture):
    def spline_sum(x):
        return jnp.sum(spline_fixture(x))

    grad_fn = jax.grad(spline_sum)
    x_test = jnp.linspace(0, 1, 10)
    grad_val = grad_fn(x_test)
    assert grad_val.shape == x_test.shape


def test_cubic_spline_vmap(spline_fixture):
    x_batch = jnp.stack([jnp.linspace(0, 1, 10), jnp.linspace(0, 1, 10) + 0.1])
    vmap_fn = eqx.filter_vmap(spline_fixture)
    y_batch = vmap_fn(x_batch)
    assert y_batch.shape == x_batch.shape


def test_cubic_spline_jit(spline_fixture):
    x_test = jnp.linspace(0, 1, 10)
    jit_fn = eqx.filter_jit(spline_fixture)
    y_jit = jit_fn(x_test)
    y_ref = spline_fixture(x_test)
    assert jnp.allclose(y_jit, y_ref)
