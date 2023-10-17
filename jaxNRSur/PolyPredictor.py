import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import equinox as eqx


class polypredictor(eqx.Module):
    coefs: Float[Array, str("n_sum")]
    bfOrders: Float[Array, str("n_sum n_lambda")]
    n_max: Int

    def __init__(
        self,
        coefs: Float[Array, str("n_sum")],
        bfOrders: Float[Array, str("n_sum n_lambda")],
        n_max: Int,
    ):
        super().__init__()

        n_node = bfOrders.shape[0]
        n_lambda = bfOrders.shape[1]
        self.coefs = jnp.zeros((n_max,))
        self.bfOrders = jnp.zeros((n_max, n_lambda))
        self.coefs = self.coefs.at[:n_node].set(coefs)
        self.bfOrders = self.bfOrders.at[:n_node].set(bfOrders)
        self.n_max = n_max

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
