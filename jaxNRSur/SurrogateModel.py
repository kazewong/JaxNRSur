import jax.numpy as jnp
import jax
from jaxNRSur.Spline import CubicSpline
from jaxNRSur.DataLoader import SurrogateDataLoader
from jaxNRSur.Harmonics import SpinWeightedSphericalHarmonics
from jaxtyping import Array, Float, Int
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
    mode_no22: list[dict]
    harmonics: list[SpinWeightedSphericalHarmonics]
    mode_22_index: int
    m_mode: Int[Array, str("n_modes-1")]

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
        self.harmonics = []
        for mode in modelist:
            if mode != (2, 2):
                self.harmonics.append(
                    SpinWeightedSphericalHarmonics(-2, mode[0], mode[1])
                )
        self.mode_no22 = [
            self.data.modes[i] for i in range(len(self.data.modes)) if i != 0
        ]
        self.mode_22_index = int(
            jnp.where((jnp.array(modelist) == jnp.array([[2, 2]])).all(axis=1))[0][0]
        )
        self.m_mode = jnp.array(
            [modelist[i][1] for i in range(len(modelist)) if i != self.mode_22_index]
        )

    @property
    def n_modes(self) -> int:
        return len(self.data.modes)

    @staticmethod
    def get_eim(
        eim_dict: dict, params: Float[Array, str("n_dim")]
    ) -> Float[Array, str("n_sample")]:
        result = jnp.zeros((eim_dict["n_nodes"], 1))
        for i in range(eim_dict["n_nodes"]):
            result = result.at[i].set(eim_dict["predictors"][i](params))
        return jnp.dot(eim_dict["eim_basis"].T, result[:, 0])

    @staticmethod
    def get_real_imag(
        mode: dict, params: Float[Array, str("n_dim")]
    ) -> tuple[Float[Array, str("n_sample")], Float[Array, str("n_sample")]]:
        params = params[None]
        real = SurrogateModel.get_eim(mode["real"], params)
        imag = SurrogateModel.get_eim(mode["imag"], params)
        return real, imag

    @staticmethod
    def get_multi_real_imag(
        modes: list[dict], params: Float[Array, str("n_dim")]
    ) -> tuple[
        list[Float[Array, str("n_sample")]], list[Float[Array, str("n_sample")]]
    ]:
        return jax.tree_util.tree_map(
            lambda mode: __class__.get_real_imag(mode, params),
            modes,
            is_leaf=lambda x: isinstance(x, dict),
        )

    def get_mode(
        self,
        real: Float[Array, str("n_sample")],
        imag: Float[Array, str("n_sample")],
        time: Float[Array, str("n_time")],
        m_mode: int,
        orbital_phase: float = 0.0,
    ) -> Float[Array, str("n_sample")]:
        return (
            CubicSpline(self.data.sur_time, real)(time)
            + 1j * CubicSpline(self.data.sur_time, imag)(time)
        ) * jnp.exp(1j * m_mode * orbital_phase)

    def get_22_mode(
        self,
        time: Float[Array, str("n_samples")],
        params: Float[Array, str("n_dim")],
        theta: float = 0.0,
    ) -> Float[Array, str("n_sample")]:
        # 22 mode has weird dict that making a specical function is easier.
        q = params[0]
        params = params[None]
        amp = self.get_eim(self.data.modes[self.mode_22_index]["amp"], params)
        phase = -self.get_eim(self.data.modes[self.mode_22_index]["phase"], params)
        phase = phase + get_T3_phase(q, self.data.sur_time)  # type: ignore
        amp_interp = CubicSpline(self.data.sur_time, amp)(time)
        phase_interp = CubicSpline(self.data.sur_time, phase)(time)
        return (
            amp_interp
            * jnp.exp(1j * phase_interp)
            * SpinWeightedSphericalHarmonics(-2, 2, 2)(theta, 0)
        )

    def get_waveform(
        self,
        time: Float[Array, str("n_sample")],
        params: Float[Array, str("n_dim")],
        theta: float = 0.0,
    ) -> Float[Array, str("n_sample")]:
        """
        Current implementation sepearates the 22 mode from the rest of the modes,
        because of the data strucutre and how they are combined.
        This means the CubicSpline is called in a loop,
        which is not ideal (double the run time).
        We should merge the datastructure to make this more efficient.
        """
        coeff = jnp.stack(jnp.array(self.get_multi_real_imag(self.mode_no22, params)))
        modes = eqx.filter_vmap(self.get_mode, in_axes=(0, 0, None, 0, None))(
            coeff[:, 0], coeff[:, 1], time, self.m_mode, theta
        )

        waveform = jnp.zeros_like(time, dtype=jnp.complex64)

        waveform += self.get_22_mode(time, params, theta)
        for i, harmonics in enumerate(self.harmonics):
            waveform += modes[i] * harmonics(theta, 0)
        return waveform
