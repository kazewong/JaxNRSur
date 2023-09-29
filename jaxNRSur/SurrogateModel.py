import jax
import jax.numpy as jnp
from jaxNRSur.Spline import CubicSpline
from jaxNRSur.DataLoader import SurrogateDataLoader
from jaxtyping import Array, Float


def get_T3_phase(q: float, t: Float[Array, str("n")], t_ref: float = 1000.0) -> float:
    eta = q / (1 + q) ** 2
    theta_raw = (eta * (t_ref - t) / 5) ** (-1.0 / 8)
    theta_cal = (eta * (t_ref + 1000) / 5) ** (-1.0 / 8)
    return 2.0 / (eta * theta_raw**5) - 2.0 / (eta * theta_cal**5)


def effective_spin(q: float, chi1: float, chi2: float) -> tuple[float, float]:
    eta = q / (1.0 + q) ** 2
    chi_wtAvg = (q * chi1 + chi2) / (1 + q)
    chi_hat = (chi_wtAvg - 38.0 * eta / 113.0 * (chi1 + chi2)) / (
        1.0 - 76.0 * eta / 113.0
    )
    chi_a = (chi1 - chi2) / 2.0
    return chi_hat, chi_a


class SurrogateModel:
    def __init__(
        self,
        data_path: str,
        modelist: list[tuple[int, int]] = [
            (2, 2),
            (2, 1),
            (2, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 3),
            (4, 4),
            (5, 5),
        ],
    ):
        self.data = SurrogateDataLoader(data_path, modelist=modelist)

    @staticmethod
    def get_eim(
        eim_dict: dict, params: Float[Array, str("n_dim")]
    ) -> Float[Array, str("n_sample")]:
        result = jnp.stack(
            jax.tree_util.tree_map(lambda f: f(params), eim_dict["predictors"]), axis=0
        )
        return jnp.dot(eim_dict["eim_basis"].T, result)

    def get_mode(
        self,
        time: Float[Array, str("n_samples")],
        params: Float[Array, str("n_dim")],
        mode: tuple[int, int],
        orbital_phase: float = 0.0,
    ) -> Float[Array, str("n_sample")]:
        if mode not in list(self.data.modes):
            raise ValueError(f"Mode {mode} not in modelist")
        elif mode == (2, 2):
            q = params[0]
            assert isinstance(q, float)
            amp = self.get_eim(self.data.modes[mode]["amp"], params)
            phase = -self.get_eim(self.data.modes[mode]["phase"], params)
            phase = phase + get_T3_phase(q, self.data.sur_time)
            amp_interp = CubicSpline(self.data.sur_time, amp)(time)
            phase_interp = CubicSpline(self.data.sur_time, phase)(time)
            return amp_interp * jnp.exp(1j * phase_interp)
        elif mode == (2, 0):
            real = self.get_eim(self.data.modes[mode]["real"], params)
            return CubicSpline(self.data.sur_time, real)(time)
        elif mode == (3, 0):
            imag = self.get_eim(self.data.modes[mode]["imag"], params)
            return CubicSpline(self.data.sur_time, imag)(time)
        else:
            real = self.get_eim(self.data.modes[mode]["real"], params)
            imag = self.get_eim(self.data.modes[mode]["imag"], params)
            return (
                CubicSpline(self.data.sur_time, real)(time)
                + 1j * CubicSpline(self.data.sur_time, imag)(time)
            ) * jnp.exp(1j * mode[1] * orbital_phase)

    # def get_waveform(
    #     self, time: Float[Array, str("n_sample")], params: dict
    # ) -> Float[Array, str("n_sample")]:
    #     pass
