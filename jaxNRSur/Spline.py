import lineax as lx
import jax.numpy as jnp
import jax
from jaxtyping import Float, Array


class CubicSpline:
    x_grid: Float[Array, " batch"]  # input x data
    y_grid: Float[Array, " n"]  # input y data

    def __init__(self, x: Float[Array, " n"], y: Float[Array, " n"]) -> None:
        self.x_grid = x
        self.diff_x = jnp.diff(x)
        self.y_grid = y

        assert len(x) == len(y), "x and y must have the same length"
        self.coeff = self.build_rep(x, y)

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        return self.get_value(x)

    def get_value(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        bin = jnp.digitize(x, self.x_grid)
        result = (
            self.coeff[bin - 1]
            * (self.x_grid[bin] - x) ** 3
            / (6 * self.diff_x[bin - 1])
        )
        result += (
            self.coeff[bin]
            * (x - self.x_grid[bin - 1]) ** 3
            / (6 * self.diff_x[bin - 1])
        )
        result += (
            (self.y_grid[bin - 1] - self.coeff[bin - 1] * self.diff_x[bin - 1] ** 2 / 6)
            * (self.x_grid[bin] - x)
            / self.diff_x[bin - 1]
        )
        result += (
            (self.y_grid[bin] - self.coeff[bin] * self.diff_x[bin - 1] ** 2 / 6)
            * (x - self.x_grid[bin - 1])
            / self.diff_x[bin - 1]
        )
        return result

    # def get_derivative(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
    # bin = jnp.digitize(x, self.x_grid)
    # result

    @staticmethod
    def divided_difference(
        x0: Float[Array, " n"],
        x1: Float[Array, " n"],
        x2: Float[Array, " n"],
        y0: Float[Array, " n"],
        y1: Float[Array, " n"],
        y2: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        d1 = (y1 - y0) / (x1 - x0)
        d2 = (y2 - y1) / (x2 - x1)
        return (d2 - d1) / (x2 - x0)

    @staticmethod
    @jax.jit
    def build_rep(x: Float[Array, " n"], y: Float[Array, " n"]) -> Float[Array, " n"]:
        # TODO: Revise boundary condition

        """
        Build the cubic spline representation of 1D curve y = f(x)

        Following mostly the algorithm from https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation

        Args:
            x (Array): x data
            y (Array): y data

        Returns:
            Array: coefficients of the cubic spline representation
        """

        diag = jnp.zeros(len(x)) + 2.0
        diff = jnp.diff(x)
        lower_diag = diff[:-1] / (diff[:-1] + diff[1:])
        upper_diag = diff[1:] / (diff[:-1] + diff[1:])
        lower_diag = jnp.concatenate([lower_diag, jnp.ones(1)])
        upper_diag = jnp.concatenate([jnp.ones(1), upper_diag])
        operator = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)
        vector = 6.0 * __class__.divided_difference(
            x[:-2], x[1:-1], x[2:], y[:-2], y[1:-1], y[2:]
        )
        low_edge = jnp.array(
            [0]
        )  # This is assuming the second derivative at the edge is 0
        high_edge = jnp.array([0])  # same here
        vector = jnp.concatenate([low_edge, vector, high_edge])

        return lx.linear_solve(operator, vector).value
