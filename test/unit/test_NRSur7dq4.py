from jaxnrsur.NRSur7dq4 import NRSur7dq4DataLoader, NRSur7dq4Model
import numpy as np
import jax.numpy as jnp


class TestNRSur7dq4:

    def __init__(self):
        self.model = NRSur7dq4Model()
        self.data = np.load("./test_data.npz")

    def test_value_check(self):
        q = 3
        incl = 0#0.14535
        phiref = 0#2.625

        chi1 = jnp.array([0.0, 0.5, 0.0])
        chi2 = jnp.array([0.5, 0.0, 0.3])

        params = jnp.concatenate([jnp.array([q]), chi1, chi2])
        h_complex = self.model.get_waveform(jnp.linspace(0, 1, 10), params, theta=incl, phi=phiref)
        assert (h_complex == self.data).all()

    def test_interp_omega_shape(self):
        pass

    def test_wignerD_jittable(self):
        pass
