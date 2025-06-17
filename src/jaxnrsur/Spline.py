import lineax as lx
import jax.numpy as jnp
import jax
from jaxtyping import Float, Array


class CubicSpline:
    """Cubic spline interpolation for 1D data.

    Attributes:
        x_grid (Float[Array, " n_grid"]): Input x data grid.
        y_grid (Float[Array, " n_grid"]): Input y data grid.
        diff_x (Float[Array, "n_grid-1"]): Differences between consecutive x values.
        coeff (Float[Array, "n_grid"]): Spline coefficients.
    """

    x_grid: Float[Array, " batch"]  # input x data
    y_grid: Float[Array, " n"]  # input y data

    def __init__(self, x: Float[Array, " n_grid"], y: Float[Array, " n_grid"]) -> None:
        """Initializes the CubicSpline object.

        Args:
            x (Float[Array, " n_grid"]): 1D array of x data points (must be sorted).
            y (Float[Array, " n_grid"]): 1D array of y data points.
        """
        self.x_grid = x
        self.diff_x = jnp.diff(x)
        self.y_grid = y

        assert len(x) == len(y), "x and y must have the same length"
        self.coeff = self.build_rep(x, y)

    def __call__(self, x: Float[Array, " n_grid"]) -> Float[Array, " n_grid"]:
        """Evaluates the spline at the given x values.

        Args:
            x (Float[Array, " n_grid"]): Points to interpolate.

        Returns:
            Float[Array, " n_grid"]: Interpolated values.
        """
        return self.get_value(x)

    def get_value(self, x: Float[Array, " n_grid"]) -> Float[Array, " n_grid"]:
        """Computes the interpolated values at given x using the cubic spline.

        Args:
            x (Float[Array, " n_grid"]): Points to interpolate.

        Returns:
            Float[Array, " n_grid"]: Interpolated values.
        """
        # Find the interval for each x
        bin = jnp.clip(jnp.digitize(x, self.x_grid), 1, len(self.x_grid) - 1)
        # Cubic spline interpolation formula
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

    @staticmethod
    def divided_difference(
        x0: Float[Array, " n"],
        x1: Float[Array, " n"],
        x2: Float[Array, " n"],
        y0: Float[Array, " n"],
        y1: Float[Array, " n"],
        y2: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Computes the second divided difference for three points.

        Args:
            x0, x1, x2 (Float[Array, " n"]): x values.
            y0, y1, y2 (Float[Array, " n"]): y values.

        Returns:
            Float[Array, " n"]: Second divided difference.
        """
        d1 = (y1 - y0) / (x1 - x0)
        d2 = (y2 - y1) / (x2 - x1)
        return (d2 - d1) / (x2 - x0)

    @staticmethod
    @jax.jit
    def build_rep(
        x: Float[Array, " n_grid"], y: Float[Array, " n_grid"]
    ) -> Float[Array, " n_grid"]:
        """Builds the cubic spline representation of 1D curve y = f(x).

        Follows the algorithm from https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation

        Args:
            x (Float[Array, " n_grid"]): x data.
            y (Float[Array, " n_grid"]): y data.

        Returns:
            Float[Array, " n_grid"]: Coefficients of the cubic spline representation.
        """
        # Set up the tridiagonal system for spline coefficients
        diag = jnp.ones(len(x))
        diag = diag.at[1:-1].set(2.0)
        diff = jnp.diff(x)
        lower_diag = diff[:-1] / (diff[:-1] + diff[1:])
        upper_diag = diff[1:] / (diff[:-1] + diff[1:])
        lower_diag = jnp.concatenate([lower_diag, jnp.ones(1)])
        upper_diag = jnp.concatenate([jnp.ones(1), upper_diag])
        operator = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)
        vector = 6.0 * __class__.divided_difference(
            x[:-2], x[1:-1], x[2:], y[:-2], y[1:-1], y[2:]
        )
        # Natural spline boundary conditions (second derivative = 0 at edges)
        low_edge = jnp.array([0])
        high_edge = jnp.array([0])
        vector = jnp.concatenate([low_edge, vector, high_edge])

        return lx.linear_solve(operator, vector).value
