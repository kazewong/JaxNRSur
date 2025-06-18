from jaxnrsur.NRSur7dq4 import NRSur7dq4Model
import numpy as np
import jax.numpy as jnp
import jax
from jaxnrsur.PolyPredictor import stable_power, PolyPredictor

# Gradient test will fail if the jax config is not set to use 64-bit precision
jax.config.update("jax_enable_x64", True)


class TestPolyPredictor:
    def test_gradient(self):
        coefs = jnp.zeros(
            1,
        )
        bfOrders = jnp.zeros((1, 7))
        predictor = PolyPredictor(coefs, bfOrders, n_max=1)
        assert not jnp.isnan(jax.grad(predictor)(jnp.zeros((7,)))).any()

    def test_stable_power(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 2.0])
        def loss_fn(x, y):
            return jnp.sum(stable_power(x, y))
            
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        result = stable_power(x, y)
        grad_result = grad_fn(x, y)
        expected = jnp.array([1.0, 1.0, 4.0])
        expected_grad = jnp.array([[0.0, 1.0, 4.0], [0.0, 0.0, jnp.log(2.)*4.]])
        assert jnp.allclose(result, expected)
        assert jnp.allclose(grad_result[0], expected_grad[0])
        assert jnp.allclose(grad_result[1], expected_grad[1])


class TestNRSur7dq4:
    def __init__(self):
        self.model = NRSur7dq4Model()
        self.data = np.load("./test_data.npz")

    def test_value_check(self):
        q = 3
        incl = 0  # 0.14535
        phiref = 0  # 2.625

        chi1 = jnp.array([0.0, 0.5, 0.0])
        chi2 = jnp.array([0.5, 0.0, 0.3])

        params = jnp.concatenate([jnp.array([q]), chi1, chi2])
        h_complex = self.model.get_waveform(
            jnp.linspace(0, 1, 10), params, theta=incl, phi=phiref
        )
        assert jnp.allclose(h_complex, self.data["waveform"])

    def test_interp_omega_shape(self):
        pass

    def test_wignerD_jittable(self):
        pass

    def test_gradient(self):
        def loss(params):
            return jnp.sum(
                self.model.get_waveform(
                    jnp.linspace(0, 1, 10), params, theta=3 * jnp.pi / 4
                )
            ).real

        params = jnp.array([0.9, 0.1, 0.4, 0.1, 0.5, 0.1, 0.3])
        grad_fn = jax.grad(loss)
        grads = grad_fn(params)
        assert ~jnp.isnan(grads).any()
