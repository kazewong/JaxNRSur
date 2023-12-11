import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import equinox as eqx


class polypredictor(eqx.Module):
    coefs: Float[Array, " n_sum"]
    bfOrders: Float[Array, " n_sum n_lambda"]
    n_max: Int

    def __init__(
        self,
        coefs: Float[Array, " n_sum"],
        bfOrders: Float[Array, " n_sum n_lambda"],
        n_max: Int,
    ):
        super().__init__()

        n_node = bfOrders.shape[0]
        self.n_max = max([n_max, n_node])
        n_lambda = bfOrders.shape[1]
        self.coefs = jnp.zeros((self.n_max,))
        self.bfOrders = jnp.zeros((self.n_max, n_lambda))
        self.coefs = self.coefs.at[:n_node].set(coefs)
        self.bfOrders = self.bfOrders.at[:n_node].set(bfOrders)

    @staticmethod
    def predict(
        X: Float[Array, " n_lambda n_sample"],
        coefs: Float[Array, " n_sum"],
        bfOrders: Float[Array, " n_sum n_lambda"],
    ):
        return jnp.dot(
            coefs,
            jnp.prod(
                jnp.power(X[jnp.newaxis, :, :], bfOrders[:, :, jnp.newaxis]),
                axis=1,
            ),
        )

    # TODO think about padding the arrays to allow for vmapping on the function
    def __call__(
        self, X: Float[Array, " n_lambda n_sample"]
    ) -> Float[Array, " n_sample"]:
        return self.predict(X, self.coefs, self.bfOrders)

    @property
    def n_nodes(self) -> int:
        return self.coefs.shape[0]
