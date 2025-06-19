import pytest
import jax.numpy as jnp
from jax import jit
from jaxnrsur.Harmonics import (
    fac,
    Cslm,
    s_lambda_lm,
    SpinWeightedSphericalHarmonics,
)


def test_fac_basic():
    assert fac(0) == 1
    assert fac(1) == 1
    assert fac(5) == 120


def test_Cslm_known_values():
    val = Cslm(0, 2, 1)
    assert jnp.isclose(val, jnp.sqrt(5.0))
    val2 = Cslm(1, 3, 2)
    assert jnp.isfinite(val2)


def test_s_lambda_lm_l_eq_m():
    # Should match the normalization for l=m=s=0
    result = s_lambda_lm(0, 0, 0, jnp.array([0.5]))
    expected = jnp.sqrt(1 / (4 * jnp.pi))
    assert jnp.allclose(result, expected)


def test_s_lambda_lm_recursive():
    # Test for l > m
    result = s_lambda_lm(0, 2, 1, jnp.array([0.0]))
    assert jnp.isfinite(result).all()


def test_spin_weighted_spherical_harmonics_basic():
    sph = SpinWeightedSphericalHarmonics(0, 1, 0)
    val = sph(jnp.pi / 2, 0.0)
    assert jnp.isfinite(val)


def test_spin_weighted_spherical_harmonics_negative_m():
    sph = SpinWeightedSphericalHarmonics(0, 2, -1)
    val = sph(jnp.pi / 3, jnp.pi / 4)
    assert jnp.isfinite(val)


def test_spin_weighted_spherical_harmonics_invalid():
    with pytest.raises(ValueError):
        SpinWeightedSphericalHarmonics(0, 1, 2)  # |m| > l


def test_spin_weighted_spherical_harmonics_jit():
    sph = SpinWeightedSphericalHarmonics(0, 2, 1)
    f = jit(sph)
    val = f(jnp.pi / 4, jnp.pi / 2)
    assert jnp.isfinite(val)
