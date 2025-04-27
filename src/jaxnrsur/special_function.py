import jax.numpy as jnp
from jax.scipy.special import gammaln

def comb(N, k):
    return jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))

def effective_spin(q: float, chi1: float, chi2: float) -> tuple[float, float]:
    eta = q / (1.0 + q) ** 2
    chi_wtAvg = (q * chi1 + chi2) / (1 + q)
    chi_hat = (chi_wtAvg - 38.0 * eta / 113.0 * (chi1 + chi2)) / (
        1.0 - 76.0 * eta / 113.0
    )
    chi_a = (chi1 - chi2) / 2.0
    return chi_hat, chi_a
