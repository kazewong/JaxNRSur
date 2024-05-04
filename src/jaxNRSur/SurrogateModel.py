from functools import partial
import jax.numpy as jnp
import jax
from jaxNRSur.Spline import CubicSpline
from jaxNRSur.DataLoader import NRHybSur3dq8DataLoader, NRSur7dq4DataLoader
from jaxNRSur.Harmonics import SpinWeightedSphericalHarmonics
from jaxNRSur.PolyPredictor import PolyPredictor, evaluate_ensemble
from jaxtyping import Array, Float, Int
import equinox as eqx

jax.config.update("jax_enable_x64", True)


def get_T3_phase(q: float, t: Float[Array, " n"], t_ref: float = 1000.0) -> float:
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


class NRHybSur3dq8Model(eqx.Module):
    data: NRHybSur3dq8DataLoader
    mode_no22: list[dict]
    harmonics: list[SpinWeightedSphericalHarmonics]
    negative_harmonics: list[SpinWeightedSphericalHarmonics]
    mode_22_index: int
    m_mode: Int[Array, " n_modes-1"]
    negative_mode_prefactor: Int[Array, " n_modes-1"]

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
        self.data = NRHybSur3dq8DataLoader(data_path, modelist=modelist)
        self.harmonics = []
        self.negative_harmonics = []
        negative_mode_prefactor = []
        for mode in modelist:
            if mode != (2, 2):
                self.harmonics.append(
                    SpinWeightedSphericalHarmonics(-2, mode[0], mode[1])
                )
                self.negative_harmonics.append(
                    SpinWeightedSphericalHarmonics(-2, mode[0], -mode[1])
                )
            if mode[1] > 0:
                negative_mode_prefactor.append((-1) ** mode[0])
            else:
                negative_mode_prefactor.append(0)

        self.mode_no22 = [
            self.data.modes[i] for i in range(len(self.data.modes)) if i != 0
        ]
        self.mode_22_index = int(
            jnp.where((jnp.array(modelist) == jnp.array([[2, 2]])).all(axis=1))[0][0]
        )
        self.m_mode = jnp.array(
            [modelist[i][1] for i in range(len(modelist)) if i != self.mode_22_index]
        )
        self.negative_mode_prefactor = jnp.array(negative_mode_prefactor)

    @property
    def n_modes(self) -> int:
        return len(self.data.modes)

    @staticmethod
    def get_eim(
        eim_dict: dict, params: Float[Array, " n_dim"]
    ) -> Float[Array, " n_sample"]:
        result = jnp.zeros((eim_dict["n_nodes"], 1))
        for i in range(eim_dict["n_nodes"]):
            result = result.at[i].set(eim_dict["predictors"][i](params))
        return jnp.dot(eim_dict["eim_basis"].T, result[:, 0])

    @staticmethod
    def get_real_imag(
        mode: dict, params: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        params = params[None]
        real = NRHybSur3dq8Model.get_eim(mode["real"], params)
        imag = NRHybSur3dq8Model.get_eim(mode["imag"], params)
        return real, imag

    @staticmethod
    def get_multi_real_imag(
        modes: list[dict], params: Float[Array, " n_dim"]
    ) -> tuple[list[Float[Array, " n_sample"]], list[Float[Array, " n_sample"]]]:
        return jax.tree_util.tree_map(
            lambda mode: __class__.get_real_imag(mode, params),
            modes,
            is_leaf=lambda x: isinstance(x, dict),
        )

    def get_mode(
        self,
        real: Float[Array, " n_sample"],
        imag: Float[Array, " n_sample"],
        time: Float[Array, " n_time"],
    ) -> Float[Array, " n_sample"]:
        return CubicSpline(self.data.sur_time, real)(time) + 1j * CubicSpline(
            self.data.sur_time, imag
        )(time)

    def get_22_mode(
        self,
        time: Float[Array, " n_samples"],
        params: Float[Array, " n_dim"],
    ) -> Float[Array, " n_sample"]:
        # 22 mode has weird dict that making a specical function is easier.
        q = params[0]
        params = params[None]
        amp = self.get_eim(self.data.modes[self.mode_22_index]["amp"], params)
        phase = -self.get_eim(self.data.modes[self.mode_22_index]["phase"], params)
        phase = phase + get_T3_phase(q, self.data.sur_time)  # type: ignore
        amp_interp = CubicSpline(self.data.sur_time, amp)(time)
        phase_interp = CubicSpline(self.data.sur_time, phase)(time)
        return amp_interp * jnp.exp(1j * phase_interp)

    def get_waveform(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: float = 0.0,
        phi: float = 0.0,
    ) -> Float[Array, " n_sample"]:
        """
        Current implementation sepearates the 22 mode from the rest of the modes,
        because of the data strucutre and how they are combined.
        This means the CubicSpline is called in a loop,
        which is not ideal (double the run time).
        We should merge the datastructure to make this more efficient.
        """
        coeff = jnp.stack(jnp.array(self.get_multi_real_imag(self.mode_no22, params)))
        modes = eqx.filter_vmap(self.get_mode, in_axes=(0, 0, None))(
            coeff[:, 0], coeff[:, 1], time
        )

        waveform = jnp.zeros_like(time, dtype=jnp.complex64)

        h22 = self.get_22_mode(time, params)
        waveform += h22 * SpinWeightedSphericalHarmonics(-2, 2, 2)(theta, phi)
        waveform += jnp.conj(h22) * SpinWeightedSphericalHarmonics(-2, 2, -2)(
            theta, phi
        )

        for i, harmonics in enumerate(self.harmonics):
            waveform += modes[i] * harmonics(theta, phi)
            waveform += (
                self.negative_mode_prefactor[i]
                * jnp.conj(modes[i])
                * self.negative_harmonics[i](theta, phi)
            )
        return waveform


class NRSur7dq4Model(eqx.Module):
    data: NRSur7dq4DataLoader
    harmonics: list[SpinWeightedSphericalHarmonics]
    negative_harmonics: list[SpinWeightedSphericalHarmonics]
    modelist_dict: dict
    n_modes: int

    def __init__(
        self,
        data_path: str,
        modelist: list[tuple[int, int]] = [
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        ],
    ):
        self.data = NRSur7dq4DataLoader(data_path, modelist=modelist)
        self.harmonics = []
        self.negative_harmonics = []

        self.n_modes = len(modelist)
        self.modelist_dict = {}
        for i in range(len(modelist)):
            self.modelist_dict[modelist[i]] = i

        for mode in modelist:
            self.harmonics.append(SpinWeightedSphericalHarmonics(-2, mode[0], mode[1]))
            self.negative_harmonics.append(
                SpinWeightedSphericalHarmonics(-2, mode[0], -mode[1])
            )

    def _get_coorb_params(
        self, q: Float, Omega: Float[Array, " n_Omega"]
    ) -> Float[Array, " n_dim"]:
        # First construct array for coorbital frame
        # borrowing notation from gwsurrogate
        coorb_x = jnp.zeros(7)

        sp = jnp.sin(Omega[4])
        cp = jnp.cos(Omega[4])

        coorb_x = coorb_x.at[0].set(q)
        coorb_x = coorb_x.at[1].set(Omega[5] * cp + Omega[6] * sp)
        coorb_x = coorb_x.at[2].set(-Omega[5] * sp + Omega[6] * cp)
        coorb_x = coorb_x.at[3].set(Omega[7])

        coorb_x = coorb_x.at[4].set(Omega[8] * cp + Omega[9] * sp)
        coorb_x = coorb_x.at[5].set(-Omega[8] * sp + Omega[9] * cp)
        coorb_x = coorb_x.at[6].set(Omega[10])

        return coorb_x

    def _get_fit_params(self, params: Float[Array, " n_dim"]) -> Float[Array, " n_dim"]:
        # Generate fit params
        fit_params = jnp.zeros(params.shape)

        q = params[0]
        eta = q / (1 + q) ** 2
        chi_wtAvg = (q * params[3] + params[6]) / (1 + q)
        chi_hat = (chi_wtAvg - 38.0 * eta / 113.0 * (params[3] + params[6])) / (
            1.0 - 76.0 * eta / 113.0
        )

        chi_a = (params[3] - params[6]) / 2

        fit_params = fit_params.at[0].set(jnp.log(q))
        fit_params = fit_params.at[1:3].set(params[1:3])
        fit_params = fit_params.at[3].set(chi_hat)
        fit_params = fit_params.at[4:6].set(params[4:6])
        fit_params = fit_params.at[6].set(chi_a)

        return fit_params

    def get_Omega_derivative_from_index(
        self,
        Omega_i: Float[Array, " n_Omega"],
        q: Float,
        predictor: PolyPredictor,
    ) -> Float[Array, " n_Omega"]:
        coorb_x = self._get_coorb_params(q, Omega_i)
        fit_params = self._get_fit_params(coorb_x)

        (
            chiA_0_fit,
            chiA_1_fit,
            chiA_2_fit,
            chiB_0_fit,
            chiB_1_fit,
            chiB_2_fit,
            omega_fit,
            omega_orb_0_fit,
            omega_orb_1_fit,
        ) = evaluate_ensemble(
            predictor, fit_params
        )  # TODO check this

        # Converting to dOmega_dt array
        dOmega_dt = jnp.zeros(len(Omega_i))

        sp = jnp.sin(Omega_i[4])
        cp = jnp.cos(Omega_i[4])

        omega_orb_x = omega_orb_0_fit * cp - omega_orb_1_fit * sp
        omega_orb_y = omega_orb_0_fit * sp + omega_orb_1_fit * cp

        # Quaterion derivative
        dOmega_dt = dOmega_dt.at[0].set(
            -0.5 * Omega_i[1] * omega_orb_x - 0.5 * Omega_i[2] * omega_orb_y
        )
        dOmega_dt = dOmega_dt.at[1].set(
            -0.5 * Omega_i[3] * omega_orb_y + 0.5 * Omega_i[0] * omega_orb_x
        )
        dOmega_dt = dOmega_dt.at[2].set(
            0.5 * Omega_i[3] * omega_orb_x + 0.5 * Omega_i[0] * omega_orb_y
        )
        dOmega_dt = dOmega_dt.at[3].set(
            0.5 * Omega_i[1] * omega_orb_y - 0.5 * Omega_i[2] * omega_orb_x
        )

        # orbital phase derivative
        dOmega_dt = dOmega_dt.at[4].set(omega_fit)

        # Spin derivatives
        dOmega_dt = dOmega_dt.at[5].set(chiA_0_fit * cp - chiA_1_fit * sp)
        dOmega_dt = dOmega_dt.at[6].set(chiA_0_fit * sp + chiA_1_fit * cp)
        dOmega_dt = dOmega_dt.at[7].set(chiA_2_fit)

        dOmega_dt = dOmega_dt.at[8].set(chiB_0_fit * cp - chiB_1_fit * sp)
        dOmega_dt = dOmega_dt.at[9].set(chiB_0_fit * sp + chiB_1_fit * cp)
        dOmega_dt = dOmega_dt.at[10].set(chiB_2_fit)

        return dOmega_dt

    def forward_euler(
        self,
        q: Float,
        Omega_i: Float[Array, " n_Omega"],
        predictor: PolyPredictor,
        dt: Float,
    ) -> Float[Array, " n_Omega"]:
        dOmega_dt = self.get_Omega_derivative_from_index(Omega_i, q, predictor)
        return Omega_i + dOmega_dt * dt

    def normalize_Omega(
        self, Omega: Float[Array, " n_Omega"], normA: float, normB: float
    ) -> Float[Array, " n_Omega"]:
        Omega_normed = jnp.zeros(len(Omega))

        nOmega = jnp.linalg.norm(Omega[:4])
        nA = jnp.linalg.norm(Omega[5:8])
        nB = jnp.linalg.norm(Omega[8:])

        Omega_normed = Omega_normed.at[:4].set(Omega[:4] / nOmega)
        Omega_normed = Omega_normed.at[5:8].set(Omega[5:8] * normA / nA)
        Omega_normed = Omega_normed.at[8:].set(Omega[8:] * normB / nB)

        return Omega_normed

    def construct_hlm_from_bases(
        self,
        lambdas: Float[Array, " n_dim"],
        predictor: PolyPredictor,
        eim_basis: Float[Array, " n_nodes n_sample"],
    ) -> Float[Array, " n_sample"]:
        return jnp.sum(
            jax.vmap(evaluate_ensemble, in_axes=(None, 0))(predictor, lambdas).T
            * eim_basis,
            axis=0,
        )

    def get_coorb_hlm(self, lambdas, mode=(2, 2)):
        # TODO bad if statement...

        idx = self.modelist_dict[mode]
        if mode[1] != 0:
            h_lm_plus = self.construct_hlm_from_bases(
                lambdas,
                self.data.modes[idx]["real_plus"]["predictors"],
                self.data.modes[idx]["real_plus"]["eim_basis"],
            ) + 1j * self.construct_hlm_from_bases(
                lambdas,
                self.data.modes[idx]["imag_plus"]["predictors"],
                self.data.modes[idx]["imag_plus"]["eim_basis"],
            )

            h_lm_minus = self.construct_hlm_from_bases(
                lambdas,
                self.data.modes[idx]["real_minus"]["predictors"],
                self.data.modes[idx]["real_minus"]["eim_basis"],
            ) + 1j * self.construct_hlm_from_bases(
                lambdas,
                self.data.modes[idx]["imag_minus"]["predictors"],
                self.data.modes[idx]["imag_minus"]["eim_basis"],
            )

            h_lm = h_lm_plus + h_lm_minus
            h_lnegm = h_lm_plus - h_lm_minus

            return h_lm, h_lnegm

        else:
            h_lm = self.construct_hlm_from_bases(
                lambdas,
                self.data.modes[idx]["real"]["predictors"],
                self.data.modes[idx]["real"]["eim_basis"],
            ) + 1j * self.construct_hlm_from_bases(
                lambdas,
                self.data.modes[idx]["imag"]["predictors"],
                self.data.modes[idx]["imag"]["eim_basis"],
            )

            return h_lm, jnp.zeros(h_lm.shape)

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 1))
    def interp_omega(
        time_grid: Float[Array, " n_grid"],
        time: Float[Array, " n_sample"],
        Omega: Float[Array, " n_Omega"],
    ):
        return CubicSpline(time_grid, Omega)(time)

    def get_waveform(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: float = 0.0,
        phi: float = 0.0,
        init_quat: Float[Array, " n_quat"] = jnp.array([1.0, 0.0, 0.0, 0.0]),
        init_orb_phase: float = 0.0,
    ) -> Float[Array, " n_sample"]:
        # TODO set up the appropriate t_low etc

        # Initialize Omega with structure:
        # Omega = [Quaterion, Orb phase, spin_1, spin_2]
        # Note that the spins are in the coprecessing frame
        q = params[0]
        Omega_0 = jnp.concatenate([init_quat, jnp.array([init_orb_phase]), params[1:]])

        normA = jnp.linalg.norm(params[1:4])
        normB = jnp.linalg.norm(params[4:7])

        init_state = (Omega_0, q, normA, normB)

        predictors_parameters, n_max = eqx.partition(self.data.coorb, eqx.is_array)
        dt = self.data.diff_t_ds

        extras = (predictors_parameters, dt)

        def timestepping_kernel(
            carry: tuple[Float[Array, " n_Omega"], Float, Float, Float], data
        ) -> tuple[
            tuple[Float[Array, " n_Omega"], Float, Float, Float],
            Float[Array, " n_Omega"],
        ]:
            Omega, q, normA, normB = carry
            predictors_parameters, dt = data
            predictor = eqx.combine(predictors_parameters, n_max)
            Omega = self.normalize_Omega(
                self.forward_euler(q, Omega, predictor, dt), normA, normB
            )
            return (Omega, q, normA, normB), Omega

        # integral timestepper
        state, Omega = jax.lax.scan(timestepping_kernel, init_state, extras)
        Omega = jnp.concatenate([Omega_0[None], Omega], axis=0)

        # Interpolating to the coorbital time array
        Omega_interp = jnp.zeros((len(self.data.t_coorb), len(Omega_0)))
        Omega_interp = self.interp_omega(self.data.t_ds, self.data.t_coorb, Omega).T

        # Get the lambda parameters to go into the waveform calculation
        lambdas = jax.vmap(self._get_fit_params)(
            jax.vmap(self._get_coorb_params, in_axes=(None, 0))(q, Omega_interp)
        )

        # loop over modes to construct the coprecessing mode array

        # TODO need to work out how to vmap this later
        copre_array = []
        coorb_h_pos = jnp.array(
            []
        )  # TODO this is just to get it to pass the precommit...
        for mode in self.modelist_dict.keys():
            # get the coorb hlms
            coorb_h_pos, coorb_h_neg = self.get_coorb_hlm(lambdas, mode=mode)

            # rotate to coprecessing frame
            print(coorb_h_neg)
            # copre_h_pos = coorb_h_pos * jnp.exp(-1j * mode[1] * Omega_interp[:, 4])
            # copre_h_neg = coorb_h_neg * jnp.exp(1j * mode[1] * Omega_interp[:, 4])

            # copre_array.append(copre_h_pos)
            # TODO store the copre

        copre_array = jnp.array(copre_array)

        # rotate with wigner D matrices

        # sum modes

        return coorb_h_pos  # lambdas, coorb_h_pos

        # coeff = jnp.stack(jnp.array(self.get_multi_real_imag(self.mode_no22, params)))
        # modes = eqx.filter_vmap(self.get_mode, in_axes=(0, 0, None))(
        #     coeff[:, 0], coeff[:, 1], time
        # )

        # waveform = jnp.zeros_like(time, dtype=jnp.complex64)

        # h22 = self.get_22_mode(time, params)
        # waveform += h22 * \
        #     SpinWeightedSphericalHarmonics(-2, 2, 2)(theta, phi)
        # waveform += jnp.conj(h22) * \
        #     SpinWeightedSphericalHarmonics(-2, 2, -2)(theta, phi)

        # for i, harmonics in enumerate(self.harmonics):
        #     waveform += modes[i] * harmonics(theta, phi)
        #     waveform += self.negative_mode_prefactor[i] * jnp.conj(modes[i]) * \
        #         self.negative_harmonics[i](theta, phi)
        # return waveform
