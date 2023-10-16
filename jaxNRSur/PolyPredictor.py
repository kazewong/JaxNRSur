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

    # TODO think about padding the arrays to allow for vmapping on the function
    def __call__(
        self, X: Float[Array, str("n_sample n_lambda")]
    ) -> Float[Array, str("n_sample")]:
        return jnp.dot(
            self.coefs,
            jnp.prod(
                jnp.power(X[jnp.newaxis, :, :], self.bfOrders[:, :, jnp.newaxis]),
                axis=1,
            ),
        )

    @property
    def n_nodes(self) -> int:
        return self.coefs.shape[0]
