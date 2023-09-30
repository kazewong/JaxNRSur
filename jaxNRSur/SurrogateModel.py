import jax.numpy as jnp
from jaxNRSur.Spline import CubicSpline
from jaxNRSur.DataLoader import SurrogateDataLoader
from jaxNRSur.Harmonics import SpinWeightedSphericalHarmonics
from jaxtyping import Array, Float
import equinox as eqx


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


class SurrogateModel(eqx.Module):
    data: SurrogateDataLoader
    harmonics: dict[tuple[int, int], SpinWeightedSphericalHarmonics]
    mode_22_index: int

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
        self.harmonics = {}
        for mode in modelist:
            self.harmonics[mode] = SpinWeightedSphericalHarmonics(-2, mode[0], mode[1])
        self.mode_22_index = int(
            jnp.where((jnp.array(modelist) == jnp.array([[2, 2]])).all(axis=1))[0][0]
        )

    @staticmethod
    def get_eim(
        eim_dict: dict, params: Float[Array, str("n_dim")]
    ) -> Float[Array, str("n_sample")]:
        result = jnp.zeros((eim_dict["n_nodes"], 1))
        for i in range(eim_dict["n_nodes"]):
            result = result.at[i].set(eim_dict["predictors"][i](params))
        return jnp.dot(eim_dict["eim_basis"].T, result[:, 0])

    def get_mode(
        self,
        time: Float[Array, str("n_samples")],
        params: Float[Array, str("n_dim")],
        mode_index: int,
        orbital_phase: float = 0.0,
    ) -> Float[Array, str("n_sample")]:
        params = params[None]
        real = self.get_eim(self.data.modes[mode_index]["real"], params)
        imag = self.get_eim(self.data.modes[mode_index]["imag"], params)

        return (
            CubicSpline(self.data.sur_time, real)(time)
            + 1j * CubicSpline(self.data.sur_time, imag)(time)
        ) * jnp.exp(1j * self.data.modes[mode_index]["mode"][1] * orbital_phase)

    def get_22_mode(
        self,
        time: Float[Array, str("n_samples")],
        params: Float[Array, str("n_dim")],
    ) -> Float[Array, str("n_sample")]:
        # 22 mode has weird dict that making a specical function is easier.
        q = params[0]
        params = params[None]
        amp = self.get_eim(self.data.modes[self.mode_22_index]["amp"], params)
        phase = -self.get_eim(self.data.modes[self.mode_22_index]["phase"], params)
        phase = phase + get_T3_phase(q, self.data.sur_time)  # type: ignore
        amp_interp = CubicSpline(self.data.sur_time, amp)(time)
        phase_interp = CubicSpline(self.data.sur_time, phase)(time)
        return amp_interp * jnp.exp(1j * phase_interp)

    # def get_waveform(
    #     self,
    #     time: Float[Array, str("n_sample")],
    #     params: Float[Array, str("n_dim")],
    #     theta: float = 0.0,
    # ) -> Float[Array, str("n_sample")]:
    #     modes = self.data.modes[1:]
    #     # Find a way to optimize it.
    #     waveform = jnp.zeros_like(time, dtype=jnp.complex64)
    #     new_dict = {key: value for key, value in self.data.modes if key != (2, 2)}

    #     waveform += self.get_22_mode(time, params) * self.harmonics[mode](theta, 0)
    #     return waveform
