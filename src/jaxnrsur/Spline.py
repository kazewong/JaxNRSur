import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


class CubicSplineFactorization(eqx.Module):
    """Reusable factorization for cubic splines on one fixed grid."""

    x_grid: Float[Array, " n_grid"]
    diff_x: Float[Array, " n_grid-1"]
    upper_diagonal: Float[Array, " n_grid-1"]
    lower_factor: Float[Array, " n_grid-1"]
    diagonal_factor: Float[Array, " n_grid"]

    def __init__(self, x: Float[Array, " n_grid"]) -> None:
        if x.ndim != 1 or len(x) < 2:
            raise ValueError(
                "x must be a one-dimensional grid with at least two points"
            )

        diff = jnp.diff(x)
        diagonal = jnp.ones(len(x), dtype=x.dtype).at[1:-1].set(2.0)
        lower = diff[:-1] / (diff[:-1] + diff[1:])
        upper = diff[1:] / (diff[:-1] + diff[1:])
        lower = jnp.concatenate([lower, jnp.ones(1, dtype=x.dtype)])
        upper = jnp.concatenate([jnp.ones(1, dtype=x.dtype), upper])

        def factor_step(previous_diagonal, values):
            lower_value, diagonal_value, upper_value = values
            multiplier = lower_value / previous_diagonal
            next_diagonal = diagonal_value - multiplier * upper_value
            return next_diagonal, (multiplier, next_diagonal)

        _, (lower_factor, diagonal_tail) = jax.lax.scan(
            factor_step,
            diagonal[0],
            (lower, diagonal[1:], upper),
        )
        self.x_grid = x
        self.diff_x = diff
        self.upper_diagonal = upper
        self.lower_factor = lower_factor
        self.diagonal_factor = jnp.concatenate([diagonal[:1], diagonal_tail])

    def solve(self, y: Float[Array, " n_grid"]) -> Float[Array, " n_grid"]:
        """Solve for spline coefficients using the stored LU factors."""
        if y.ndim != 1 or len(y) != len(self.x_grid):
            raise ValueError("y must be one-dimensional and match the spline grid")

        d1 = (y[1:-1] - y[:-2]) / self.diff_x[:-1]
        d2 = (y[2:] - y[1:-1]) / self.diff_x[1:]
        interior = 6.0 * (d2 - d1) / (self.x_grid[2:] - self.x_grid[:-2])
        vector = jnp.concatenate(
            [jnp.zeros(1, dtype=y.dtype), interior, jnp.zeros(1, dtype=y.dtype)]
        )

        def forward_step(previous_value, values):
            lower_factor, vector_value = values
            next_value = vector_value - lower_factor * previous_value
            return next_value, next_value

        _, forward_tail = jax.lax.scan(
            forward_step,
            vector[0],
            (self.lower_factor, vector[1:]),
        )
        forward = jnp.concatenate([vector[:1], forward_tail])
        last = forward[-1] / self.diagonal_factor[-1]

        def backward_step(next_value, values):
            upper_value, diagonal_value, forward_value = values
            value = (forward_value - upper_value * next_value) / diagonal_value
            return value, value

        _, reversed_head = jax.lax.scan(
            backward_step,
            last,
            (
                self.upper_diagonal[::-1],
                self.diagonal_factor[:-1][::-1],
                forward[:-1][::-1],
            ),
        )
        return jnp.concatenate([reversed_head[::-1], last[None]])


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

    def __init__(
        self,
        x: Float[Array, " n_grid"],
        y: Float[Array, " n_grid"],
        factorization: CubicSplineFactorization | None = None,
    ) -> None:
        """Initializes the CubicSpline object.

        Args:
            x (Float[Array, " n_grid"]): 1D array of x data points (must be sorted).
            y (Float[Array, " n_grid"]): 1D array of y data points.
            factorization (CubicSplineFactorization, optional): Reusable
                factorization for the same x grid.
        """
        self.x_grid = x
        self.diff_x = jnp.diff(x)
        self.y_grid = y

        assert len(x) == len(y), "x and y must have the same length"
        if factorization is None:
            self.coeff = self.build_rep(x, y)
        else:
            self.coeff = factorization.solve(y)

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
        return CubicSplineFactorization(x).solve(y)
