import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx


class polypredictor(eqx.Module):
    coefs: Float[Array, str("n_sum")]
    bfOrders: Float[Array, str("n_sum n_lambda")]

    def __init__(
        self,
        coefs: Float[Array, str("n_sum")],
        bfOrders: Float[Array, str("n_sum n_lambda")],
    ):
        super().__init__()

        self.coefs = coefs
        self.bfOrders = bfOrders

    def __call__(
        self, X: Float[Array, str("n_sample")]
    ) -> Float[Array, str("n_sample")]:
        return jnp.dot(
            self.coefs,
            jnp.prod(
                jnp.power(X[jnp.newaxis, :, :], self.bfOrders[:, :, jnp.newaxis]),
                axis=1,
            ),
        )
