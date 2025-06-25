import pytest
import jax.numpy as jnp
from jaxnrsur.jaxnrsur import JaxNRSur, WaveformModel, DataLoader, RSUN_SI, MPC_SI, C_SI
from jaxnrsur.NRSur7dq4 import NRSur7dq4Model
from jaxnrsur.NRHybSur3dq8 import NRHybSur3dq8Model
from jaxtyping import Float, Array

# INFO: This test suites is mostly generated with AI and edited by Kaze. W. K. Wong.


# Dummy DataLoader for testing
class DummyDataLoader(DataLoader):
    def __init__(self):
        self.sur_time = jnp.linspace(0, 100, 1000)
        self.modes = [(2, 2), (2, 1), (2, 0)]


# Dummy WaveformModel for testing
class DummyWaveformModel(WaveformModel):
    def __init__(self):
        self.data = DummyDataLoader()

    def get_waveform_geometric(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_param"],
        theta: float,
        phi: float,
    ):
        # Return simple sinusoids for testing
        return jnp.sin(time), jnp.cos(time)


@pytest.fixture
def jaxnrsur_instance():
    model = DummyWaveformModel()
    jaxnrsur = JaxNRSur(model=model, alpha_window=0.1)
    return jaxnrsur


def test_get_waveform_td_basic(jaxnrsur_instance):
    time = jnp.linspace(0, 1, 100)
    params = jnp.array([30.0, 100.0, 0.5, 0.2])  # mtot, dist_mpc, dummy params
    hp, hc = jaxnrsur_instance.get_waveform_td(time, params)
    assert hp.shape == time.shape
    assert hc.shape == time.shape
    # Check scaling: output should be proportional to mtot/ dist_mpc, including window
    mtot = params[0]
    dist_mpc = params[1]
    const = mtot * RSUN_SI / dist_mpc / MPC_SI
    time_m = time * C_SI / RSUN_SI / mtot

    # Reproduce the window function from JaxNRSur
    Tcoorb = (
        jaxnrsur_instance.model.data.sur_time[-1]
        - jaxnrsur_instance.model.data.sur_time[0]
    )
    window_start = jnp.max(
        jnp.array([time_m[0], jaxnrsur_instance.model.data.sur_time[0]])
    )
    window_end = window_start + jaxnrsur_instance.alpha_window * Tcoorb
    x = (time_m - window_start) / (window_end - window_start)
    window = jnp.select(
        [time_m < window_start, time_m > window_end],
        [0.0, 1.0],
        default=x * x * x * (10 + x * (6 * x - 15)),
    )

    expected_hp = jnp.sin(time_m) * window * const
    expected_hc = jnp.cos(time_m) * window * const

    assert jnp.allclose(hp, expected_hp, atol=1e-6)
    assert jnp.allclose(hc, expected_hc, atol=1e-6)


def test_get_waveform_td_windowing(jaxnrsur_instance):
    # Test that windowing is applied (alpha_window > 0)
    jaxnrsur_instance.alpha_window = 0.2
    time = jnp.linspace(0, 1, 100)
    params = jnp.array([30.0, 100.0, 0.5, 0.2])
    hp, hc = jaxnrsur_instance.get_waveform_td(time, params)
    # The start of the waveform should be zero due to windowing
    assert jnp.isclose(hp[0], 0.0, atol=1e-8)
    assert jnp.isclose(hc[0], 0.0, atol=1e-8)


def test_get_waveform_fd_basic(jaxnrsur_instance):
    time = jnp.linspace(0, 1, 128)
    params = jnp.array([30.0, 100.0, 0.5, 0.2])
    hp_fd, hc_fd = jaxnrsur_instance.get_waveform_fd(time, params)
    # Output should be half the length + 1 (rfft style)
    assert hp_fd.shape[0] == 65
    assert hc_fd.shape[0] == 65
    # Should be complex arrays
    assert jnp.iscomplexobj(hp_fd)
    assert jnp.iscomplexobj(hc_fd)


def test_nrsur7dq4model_waveform():
    model = NRSur7dq4Model()
    jaxnrsur = JaxNRSur(model=model, alpha_window=0.1)
    # Just check that the model runs and returns arrays of correct shape
    time = jnp.linspace(0, 1, 100)
    params = jnp.array([30.0, 100.0, 3., 0.5, 0.0, 0.0, 0.0, 0.5, 0.0])  # mtot, dist_mpc, chi1z, chi2z, theta1, phi1, theta2, phi2
    hp, hc = jaxnrsur.get_waveform_td(time, params)
    assert hp.shape == time.shape
    assert hc.shape == time.shape
    
def test_nrhybsur3dq8model_waveform():
    model = NRHybSur3dq8Model()
    jaxnrsur = JaxNRSur(model=model, alpha_window=0.1)
    # Just check that the model runs and returns arrays of correct shape
    time = jnp.linspace(0, 1, 100)
    params = jnp.array([30.0, 100.0, 0.9, 0.1, 0.1])  # mtot, dist_mpc, q, chi1z, chi2z
    hp, hc = jaxnrsur.get_waveform_td(time, params)
    assert hp.shape == time.shape
    assert hc.shape == time.shape


def test_waveformmodel_not_implemented():
    class IncompleteModel(WaveformModel):
        pass

    model = IncompleteModel()
    with pytest.raises(NotImplementedError):
        model.get_waveform_geometric(jnp.array([0.0]), jnp.array([0.0]), 0.0, 0.0)
