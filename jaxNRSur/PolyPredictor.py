import jax.numpy as jnp
from jaxtyping import Array, Float, Int, jaxtyped
import equinox as eqx


class PolyPredictor(eqx.Module):
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

    @jaxtyped
    @staticmethod
    def predict(
        inputs: Float[Array, " n_lambda"],
        coefs: Float[Array, " n_sum"],
        bfOrders: Float[Array, " n_sum n_lambda"],
    ) -> Float[Array, " 1"]:
        # TODO: Check this Ethan
        return jnp.dot(
            coefs,
            jnp.prod(
                jnp.power(inputs, bfOrders),
                axis=1,
            ),
        )

    def __call__(self, X: Float[Array, " n_lambda "]) -> Float[Array, " 1"]:
        return self.predict(X, self.coefs, self.bfOrders)

    @property
    def n_nodes(self) -> int:
        return self.coefs.shape[0]


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_ensemble(
    predictors: PolyPredictor, inputs: Float[Array, " n_lambda"]
) -> Float[Array, " n_predictor"]:
    return predictors(inputs)


@eqx.filter_vmap(in_axes=(0, 0, None))
def make_polypredictor_ensemble(
    coefs: Float[Array, " n_predictor n_sum"],
    bfOrders: Float[Array, " n_predictor n_sum n_lambda"],
    n_max: int,
):
    return PolyPredictor(coefs, bfOrders, n_max)
