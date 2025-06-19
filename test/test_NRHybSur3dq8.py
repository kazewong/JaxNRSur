import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxnrsur.NRHybSur3dq8 import NRHybSur3dq8Model


@pytest.fixture(scope="module")
def model():
    # Use default modelist for basic test
    return NRHybSur3dq8Model()


@pytest.fixture(scope="module")
def time():
    # Reasonable time array for waveform
    return jnp.linspace(-1000, 100, 1000)


@pytest.fixture(scope="module")
def params():
    # Example parameters: q, chi1z, chi2z
    return jnp.array([0.9, 0.1, 0.1])


def _check_waveform(h, shape):
    assert h.shape == shape
    assert jnp.iscomplexobj(h)
    assert not jnp.isnan(h).any()


@pytest.mark.parametrize(
    "waveform_fn, param_shape",
    [
        (lambda m, t, p: m(t, p), (1000,)),  # basic
        (
            lambda m, t, p: eqx.filter_jit(
                eqx.filter_vmap(m.get_waveform, in_axes=(None, 0))
            )(t, jnp.repeat(p[None, :], 5, axis=0)),
            (5, 1000),
        ),  # jit+vmap
    ],
)
def test_waveform_variants(model, time, params, waveform_fn, param_shape):
    h = waveform_fn(model, time, params)
    _check_waveform(h, param_shape)


def test_model_initialization(model):
    assert isinstance(model, NRHybSur3dq8Model)
    assert model.n_modes > 0
    assert hasattr(model, "data")
    assert hasattr(model, "harmonics")


def test_grad_waveform_time(model, time, params):
    # Gradient with respect to time
    def target(time_):
        return jnp.sum(model(time_, params)).real

    grad_time = jax.grad(target)
    grad_val = grad_time(time)
    assert grad_val.shape == time.shape
    assert not jnp.isnan(grad_val).any()


def test_grad_waveform_params(model, time, params):
    # Gradient with respect to params
    def target(params_):
        return jnp.sum(model(time, params_)).real

    grad_params = jax.grad(target)
    grad_val = grad_params(params)
    assert grad_val.shape == params.shape
    assert not jnp.isnan(grad_val).any()
