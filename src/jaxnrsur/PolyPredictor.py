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
        return jnp.dot(
            coefs,
            jnp.prod(
                jnp.power(inputs, bfOrders),
                axis=1,
            ),
        )

    def __call__(self, X: Float[Array, " n_lambda "]) -> Float[Array, " 1"]:
        return self.predict(X, self.coefs, self.bfOrders)
    
    def predict_at_index(self, X: Float[Array, " n_lambda "], idx: Int) -> Float[Array, " 1"]:
        return self.predict(X, self.coefs[idx], self.bfOrders[idx])

    @property
    def n_nodes(self) -> int:
        return self.coefs.shape[0]

@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_ensemble_dynamics(
    predictors: PolyPredictor, inputs: Float[Array, " n_lambda"]
) -> Float[Array, " n_predictor"]:
    return predictors(inputs)

@eqx.filter_vmap(in_axes=(None, eqx.if_array(0))) # TODO this is not working. I need something where I can specify the index and iterate over all the specific output parameters 
def evaluate_ensemble_dynamics_at_index(
    predictors: PolyPredictor, inputs: Float[Array, " n_lambda"], index: int
) -> Float[Array, " n_predictor"]:
    return predictors.predict_at_index(inputs, predictors.coefs, predictors.bfOrders, index)

@eqx.filter_vmap(in_axes=(eqx.if_array(0), eqx.if_array(0)))
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
