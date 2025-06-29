from jaxnrsur.NRSur7dq4 import NRSur7dq4Model
import numpy as np
import jax.numpy as jnp
import jax
from jaxnrsur.PolyPredictor import stable_power, PolyPredictor
import equinox as eqx

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
        expected_grad = jnp.array([[0.0, 1.0, 4.0], [0.0, 0.0, jnp.log(2.0) * 4.0]])
        assert jnp.allclose(result, expected)
        assert jnp.allclose(grad_result[0], expected_grad[0])
        assert jnp.allclose(grad_result[1], expected_grad[1])


class TestNRSur7dq4:
    def __init__(self):
        self.model = NRSur7dq4Model()
        self.data = np.load("./test_data.npz")

    def test_gradient(self):
        def loss(params):
            hp, hc = self.model.get_waveform_geometric(
                jnp.linspace(0, 1, 10), params, theta=3 * jnp.pi / 4
            )
            return jnp.sum(hp**2 + hc**2)

        params = jnp.array([0.9, 0.1, 0.4, 0.1, 0.5, 0.1, 0.3])
        grad_fn = jax.grad(loss)
        grads = grad_fn(params)
        assert ~jnp.isnan(grads).any()

    def test_waveform_agreement(self):
        """
        Cross-check: Compare the JAX model waveform with reference data.
        """
        # Reference waveform loaded from file
        test_data = np.load("./test_data.npz", allow_pickle=True)
        h_ref = test_data["h"]  # shape: (2, N) or (N, 2) depending on convention
        theta_ref = test_data["incl"]
        phi_ref = test_data["phiref"]

        # Prepare inputs for JAX model
        params = test_data["params"]

        model = self.model

        for i in range(len(params)):
            sample_params = params[i]
            inertial_h = eqx.filter_jit(model.get_waveform_inertial_permode)(
                sample_params,
                theta=theta_ref[i],
                phi=phi_ref[i],
            )
            inertial_h = jnp.sum(inertial_h, axis=1)  # Sum over modes

            print(
                f"Sample {i}: Max abs difference in hp:",
                np.max(np.abs(h_ref[i] - inertial_h)),
            )
            assert np.max(np.abs(h_ref[i] - inertial_h)) < 1e-4, (
                f"Waveform mismatch for sample {i}: "
                f"Max abs difference {np.max(np.abs(h_ref[i] - inertial_h))}"
            )
