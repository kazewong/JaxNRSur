import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped
import equinox as eqx
import jax


@jax.custom_jvp
def stable_power(
    x: Float[Array, " n"], y: Float[Array, "... n"]
) -> Float[Array, "... n"]:
    """Elementwise power with custom JVP for stability.

    Args:
        x (Float[Array, " n"]): Base array.
        y (Float[Array, "... n"]): Exponent array.

    Returns:
        Float[Array, "... n"]: x raised to the power y, elementwise.
    """
    return jnp.power(x, y)


@stable_power.defjvp
def stable_power_jvp(primals, tangents):
    """Custom JVP rule for stable_power to handle edge cases and differentiation."""
    x, y = primals
    x_dot, y_dot = tangents

    primal_out = stable_power(x, y)
    tangent_out = (
        jnp.where((x == 0) & (y == 0), jnp.zeros(y.shape), y * jnp.power(x, y - 1))
        * x_dot
        + jnp.where(
            (x == 0) & (y == 0), jnp.zeros(x.shape), jnp.log(x) * jnp.power(x, y)
        )
        * y_dot
    )

    return primal_out, tangent_out


class PolyPredictor(eqx.Module):
    """Polynomial predictor using basis functions and coefficients.

    Attributes:
        coefs (Float[Array, " n_sum"]): Coefficient array.
        bfOrders (Float[Array, " n_sum n_lambda"]): Basis function orders.
        n_max (Int): Maximum number of nodes.
    """

    coefs: Float[Array, " n_sum"]
    bfOrders: Float[Array, " n_sum n_lambda"]
    n_max: Int

    def __init__(
        self,
        coefs: Float[Array, " n_sum"],
        bfOrders: Float[Array, " n_sum n_lambda"],
        n_max: Int,
    ):
        """Initializes PolyPredictor with coefficients and basis function orders.

        Args:
            coefs (Float[Array, " n_sum"]): Coefficient array.
            bfOrders (Float[Array, " n_sum n_lambda"]): Basis function orders.
            n_max (Int): Maximum number of nodes.
        """
        super().__init__()

        n_node = bfOrders.shape[0]
        self.n_max = max([n_max, n_node])
        n_lambda = bfOrders.shape[1]
        self.coefs = jnp.zeros((self.n_max,))
        self.bfOrders = jnp.zeros((self.n_max, n_lambda))
        self.coefs = self.coefs.at[:n_node].set(coefs)
        self.bfOrders = self.bfOrders.at[:n_node].set(bfOrders)

    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def predict(
        inputs: Float[Array, " n_lambda"],
        coefs: Float[Array, " n_sum"],
        bfOrders: Float[Array, " n_sum n_lambda"],
    ) -> Float[Array, "..."]:
        """Predicts output using polynomial basis functions.

        Args:
            inputs (Float[Array, " n_lambda"]): Input array.
            coefs (Float[Array, " n_sum"]): Coefficient array.
            bfOrders (Float[Array, " n_sum n_lambda"]): Basis function orders.

        Returns:
            Float[Array, "..."]: Predicted output.
        """
        return jnp.dot(
            coefs,
            jnp.prod(
                stable_power(inputs, jax.lax.stop_gradient(bfOrders)),
                axis=1,
            ),
        )

    def __call__(self, X: Float[Array, " n_lambda "]) -> Float[Array, " 1"]:
        """Calls the predictor on input X.

        Args:
            X (Float[Array, " n_lambda "]): Input array.

        Returns:
            Float[Array, " 1"]: Predicted output.
        """
        return self.predict(X, self.coefs, self.bfOrders)

    def predict_at_index(
        self, X: Float[Array, " n_lambda "], idx: Int
    ) -> Float[Array, " 1"]:
        """Predicts output at a specific index.

        Args:
            X (Float[Array, " n_lambda "]): Input array.
            idx (Int): Index of the coefficient and basis function order to use.

        Returns:
            Float[Array, " 1"]: Predicted output at the given index.
        """
        return self.predict(X, self.coefs[idx], self.bfOrders[idx])

    @property
    def n_nodes(self) -> int:
        """Returns the number of nodes (nonzero coefficients)."""
        return self.coefs.shape[0]


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_ensemble_dynamics(
    predictors: PolyPredictor, inputs: Float[Array, " n_lambda"]
) -> Float[Array, " n_predictor"]:
    """Evaluates an ensemble of predictors on the same input.

    Args:
        predictors (PolyPredictor): Ensemble of predictors.
        inputs (Float[Array, " n_lambda"]): Input array.

    Returns:
        Float[Array, " n_predictor"]: Ensemble predictions.
    """
    return predictors(inputs)


@eqx.filter_vmap(in_axes=(eqx.if_array(0), eqx.if_array(0)))
def evaluate_ensemble(
    predictors: PolyPredictor, inputs: Float[Array, " n_lambda"]
) -> Float[Array, " n_predictor"]:
    """Evaluates an ensemble of predictors, each on its own input.

    Args:
        predictors (PolyPredictor): Ensemble of predictors.
        inputs (Float[Array, " n_lambda"]): Input array for each predictor.

    Returns:
        Float[Array, " n_predictor"]: Ensemble predictions.
    """
    return predictors(inputs)


@eqx.filter_vmap(in_axes=(0, 0, None))
def make_polypredictor_ensemble(
    coefs: Float[Array, " n_predictor n_sum"],
    bfOrders: Float[Array, " n_predictor n_sum n_lambda"],
    n_max: int,
):
    """Creates an ensemble of PolyPredictor objects.

    Args:
        coefs (Float[Array, " n_predictor n_sum"]): Coefficient arrays for each predictor.
        bfOrders (Float[Array, " n_predictor n_sum n_lambda"]): Basis function orders for each predictor.
        n_max (int): Maximum number of nodes.

    Returns:
        PolyPredictor: Ensemble of PolyPredictor objects.
    """
    return PolyPredictor(coefs, bfOrders, n_max)
