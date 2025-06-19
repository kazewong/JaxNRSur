import math
import pytest
import jax
from jaxnrsur.special_function import comb, effective_spin


class TestComb:
    @pytest.mark.parametrize(
        "N, k, expected",
        [
            (5, 2, 10.0),
            (10, 0, 1.0),
            (10, 10, 1.0),
            (6, 3, 20.0),
            (0, 0, 1.0),
        ],
    )
    def test_basic(self, N, k, expected):
        assert comb(N, k) == pytest.approx(expected)

    @pytest.mark.parametrize("N", range(0, 11))
    def test_compare_math(self, N):
        for k in range(0, N + 1):
            expected = math.comb(N, k)
            result = comb(N, k)
            assert result == pytest.approx(expected), f"comb({N},{k})"

    def test_jit(self):
        jit_comb = jax.jit(comb)
        assert jit_comb(7, 3) == pytest.approx(35.0)


class TestEffectiveSpin:
    def test_symmetric(self):
        q = 1.0
        chi1 = 0.5
        chi2 = 0.5
        chi_hat, chi_a = effective_spin(q, chi1, chi2)
        assert chi_hat == pytest.approx(chi1)
        assert chi_a == pytest.approx(0.0)

    def test_antisymmetric(self):
        q = 1.0
        chi1 = 0.7
        chi2 = -0.7
        chi_hat, chi_a = effective_spin(q, chi1, chi2)
        assert chi_hat == pytest.approx(0.0)
        assert chi_a == pytest.approx(0.7)

    def test_general(self):
        q = 2.0
        chi1 = 0.3
        chi2 = -0.2
        chi_hat, chi_a = effective_spin(q, chi1, chi2)
        assert isinstance(chi_hat, float)
        assert isinstance(chi_a, float)
        assert chi_a == pytest.approx((chi1 - chi2) / 2.0)
        assert min(chi1, chi2) - 1e-2 <= chi_hat <= max(chi1, chi2) + 1e-2

    def test_jit(self):
        jit_effective_spin = jax.jit(effective_spin)
        q = 1.5
        chi1 = 0.1
        chi2 = -0.3
        chi_hat, chi_a = jit_effective_spin(q, chi1, chi2)
        chi_hat_ref, chi_a_ref = effective_spin(q, chi1, chi2)
        assert chi_hat == pytest.approx(chi_hat_ref)
        assert chi_a == pytest.approx(chi_a_ref)
