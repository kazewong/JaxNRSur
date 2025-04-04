from functools import partial
import jax.numpy as jnp
import jax
from jax.scipy.special import factorial, gammaln
from jaxNRSur.Spline import CubicSpline
from jaxNRSur.DataLoader import NRHybSur3dq8DataLoader, NRSur7dq4DataLoader
from jaxNRSur.Harmonics import SpinWeightedSphericalHarmonics
from jaxNRSur.PolyPredictor import PolyPredictor, evaluate_ensemble, evaluate_ensemble_dynamics, make_polypredictor_ensemble
from jaxtyping import Array, Float, Int
import equinox as eqx

jax.config.update("jax_enable_x64", True)


def comb(N, k):
    return jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))


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
    mode_22_index: int
    m_mode: Int[Array, " n_modes-1"]
    negative_mode_prefactor: Int[Array, " n_modes-1"]

    def __init__(
        self,
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
        """
        Initialize NRHybSur3dq8Model.
        The model is described in the paper:
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.064045

        Args:
            modelist (list[tuple[int, int]]): List of modes to be used.
        """
        self.data = NRHybSur3dq8DataLoader(modelist=modelist)
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

    def __call__(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: float = 0.0,
        phi: float = 0.0,
    ) -> Float[Array, " n_sample"]:
        """
        Alias for get_waveform.

        Args:
            time (Float[Array, " n_sample"]): Time grid.
            params (Float[Array, " n_dim"]): Source parameters.
            theta (float, optional): Polar angle. Defaults to 0.0.
            phi (float, optional): Azimuthal angle. Defaults to 0.0.
        
        """
        return self.get_waveform(time, params, theta, phi)

    @property
    def n_modes(self) -> int:
        """
        Number of modes in the model.
        """
        return len(self.data.modes)

    @staticmethod
    def get_eim(
        eim_dict: dict, params: Float[Array, " n_dim"]
    ) -> Float[Array, " n_sample"]:
        """
        Construct the EIM basis given the source parameters.

        Args:
            eim_dict (dict): EIM dictionary.
            params (Float[Array, " n_dim"]): Source parameters.
        """
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
    modelist_dict: dict
    modelist_dict_extended: dict
    harmonics: list[SpinWeightedSphericalHarmonics]
    n_modes: int
    n_modes_extended: int

    def __init__(
        self,
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
        self.data = NRSur7dq4DataLoader(modelist=modelist)
        self.harmonics = []

        self.n_modes = len(modelist)
        self.modelist_dict = {}
        for i, mode in enumerate(modelist):
            self.modelist_dict[mode] = i

        self.modelist_dict_extended = {}
        idx = 0
        for i, mode in enumerate(modelist):
            self.modelist_dict_extended[mode] = idx
            idx += 1
            
            if mode[1] > 0:
                negative_mode = (mode[0], -mode[1])
                self.modelist_dict_extended[negative_mode] = idx
                idx += 1

        self.n_modes_extended = len(self.modelist_dict_extended.keys())

        for mode in list(self.modelist_dict_extended.keys()):
            self.harmonics.append(SpinWeightedSphericalHarmonics(-2, mode[0], mode[1]))

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

        # Transforming log(q) such that it is bounded between [-1, 1], values takes from the NRSur7dq4 code
        q_fit_offset = -0.9857019407834238
        q_fit_slope = 1.4298059216576398

        eta = q / (1 + q) ** 2
        chi_wtAvg = (q * params[3] + params[6]) / (1 + q)
        chi_hat = (chi_wtAvg - 38.0 * eta / 113.0 * (params[3] + params[6])) / (
            1.0 - 76.0 * eta / 113.0
        )

        chi_a = (params[3] - params[6]) / 2

        fit_params = fit_params.at[0].set(q_fit_offset + q_fit_slope*jnp.log(q))
        fit_params = fit_params.at[1:3].set(params[1:3])
        fit_params = fit_params.at[3].set(chi_hat)
        fit_params = fit_params.at[4:6].set(params[4:6])
        fit_params = fit_params.at[6].set(chi_a)

        return fit_params

    def get_Omega_derivative(
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
        ) = evaluate_ensemble_dynamics(
            predictor, fit_params
        )

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

    def get_Omega_derivative_from_index(
        self,
        Omega_i: Float[Array, " n_Omega"],
        q: Float,
        predictor: PolyPredictor,
        index: int
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
        ) = [predictor.predict_at_index(fit_params, index) for i in range(9)]

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
        dOmega_dt = self.get_Omega_derivative(Omega_i, q, predictor)
        return Omega_i + dOmega_dt * dt

    def AB4(
        self, 
        q: Float,
        Omega_i4: Float[Array, " 4 n_Omega"],
        k_ab4: Float[Array, " 3 n_Omega"],
        predictor: PolyPredictor,
        dt: Float[Array, " 4"],
    ) -> tuple[Float[Array, " n_Omega"], Float[Array, " n_Omega"]]:

        dOmega_dt = self.get_Omega_derivative(Omega_i4[-1], q, predictor)

        # Using the timestep variable AB4 from gwsurrogate
        dt1 = dt[0]; dt2 = dt[1]; dt3 = dt[2]; dt4 = dt[3]

        dt12 = dt1 + dt2
        dt123 = dt12 + dt3
        dt23 = dt2 + dt3

        D1 = dt1 * dt12 * dt123
        D2 = dt1 * dt2 * dt23
        D3 = dt2 * dt12 * dt3

        B41 = dt3 * dt23 / D1
        B42 = -1 * dt3 * dt123 / D2
        B43 = dt23 * dt123 / D3
        B4 = B41 + B42 + B43

        C41 = (dt23 + dt3) / D1
        C42 = -1 * (dt123 + dt3) / D2
        C43 = (dt123 + dt23) / D3
        C4 = C41 + C42 + C43

        A = dOmega_dt;
        B = dOmega_dt*B4 - k_ab4[0]*B41 - k_ab4[1]*B42 - k_ab4[2]*B43;
        C = dOmega_dt*C4 - k_ab4[0]*C41 - k_ab4[1]*C42 - k_ab4[2]*C43;
        D = (dOmega_dt-k_ab4[0])/D1 - (dOmega_dt-k_ab4[1])/D2 + (dOmega_dt-k_ab4[2])/D3

        return Omega_i4[-1] +  dt4 * (A + dt4 * (0.5*B + dt4*( C/3.0 + dt4*0.25*D))), dOmega_dt

    
    def RK4(
        self, 
        q: Float,
        Omega: Float[Array, " n_Omega"],
        predictors,
        dt: Float,
    ) -> tuple[Float[Array, " n_Omega"], Float[Array, " n_Omega"]]:

        # WARNING: the current way of running the RK4 steps is likely
        # to be inefficient. Need to adjust it.

        predictor_parameters, n_max = eqx.partition(predictors, eqx.is_array)

        def get_RK4_Omega_derivatives(
                carry: tuple[Float[Array, " n_Omega"], Float, Float[Array, " n_Omega"]],
                data: tuple[PolyPredictor, Float[Array, " 4"]]
        ):
            
            Omega, q, derivative = carry
            predictor_parameters, dt = data

            predictors = eqx.combine(predictor_parameters, n_max)
            derivative = self.get_Omega_derivative(Omega + dt * derivative, q, predictors)

            return (Omega, q, derivative), derivative
        
        state, dOmega_dt_rk4 = jax.lax.scan(get_RK4_Omega_derivatives, (Omega, q, jnp.zeros(len(Omega))), 
                                            (predictor_parameters, jnp.array([0, 1, 1, 2])*dt))

        Omega_next = Omega + (1./3.) * (dt[0] * dOmega_dt_rk4[0] + 2*dt[0] * dOmega_dt_rk4[1] + 2*dt[0] *dOmega_dt_rk4[2] + dt[0] * dOmega_dt_rk4[3])
        predictor_i = make_polypredictor_ensemble(predictors.coefs[0], predictors.bfOrders[0], 100)
        k_next = self.get_Omega_derivative(Omega, q, predictor_i)

        return Omega_next, k_next

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

        Omega_normed = Omega_normed.at[4].set(Omega[4])

        return Omega_normed

    def construct_hlm_from_bases(
        self,
        lambdas: Float[Array, " n_dim"],
        predictor: PolyPredictor,
        eim_basis: Float[Array, " n_nodes n_sample"],
    ) -> Float[Array, " n_sample"]:

        return jnp.dot(evaluate_ensemble(predictor, lambdas), eim_basis)

    def get_coorb_hlm(self, lambdas, mode=(2, 2)):

        # surrogate is built on the symmetric (sum) and antisymmetric (diff)
        # combinations of the +|m| and -|m| modes
        # (although they are confusingly labeled "plus" and "minus" in 
        # the file)
        idx = self.modelist_dict[mode]
        h_lm_sum = self.construct_hlm_from_bases(
            lambdas[self.data.modes[idx]["real_plus"]["node_indices"]],
            self.data.modes[idx]["real_plus"]["predictors"],
            self.data.modes[idx]["real_plus"]["eim_basis"],
        ) + 1j * self.construct_hlm_from_bases(
            lambdas[self.data.modes[idx]["imag_plus"]["node_indices"]],
            self.data.modes[idx]["imag_plus"]["predictors"],
            self.data.modes[idx]["imag_plus"]["eim_basis"],
        )

        h_lm_diff = self.construct_hlm_from_bases(
            lambdas[self.data.modes[idx]["real_minus"]["node_indices"]],
            self.data.modes[idx]["real_minus"]["predictors"],
            self.data.modes[idx]["real_minus"]["eim_basis"],
        ) + 1j * self.construct_hlm_from_bases(
            lambdas[self.data.modes[idx]["imag_minus"]["node_indices"]],
            self.data.modes[idx]["imag_minus"]["predictors"],
            self.data.modes[idx]["imag_minus"]["eim_basis"],
        )

        # Eq. 6 in https://arxiv.org/pdf/1905.09300.pdf
        # NOTE: factor of 2?
        # also, in m=0 case sum = diff currently
        # but it used to be zero
        # NOTE: Ethan (10/6/24), changing this code following: 
        # https://github.com/sxs-collaboration/gwsurrogate/blob/55dfadb9e62de0f1ae0c9d69c72e49b00a760d85/gwsurrogate/new/precessing_surrogate.py#L714
        h_lm_plus = jnp.conj(h_lm_sum - h_lm_diff)*(jnp.abs(h_lm_diff) > 1e-12) + h_lm_sum*(jnp.abs(h_lm_diff) < 1e-12) 
        h_lm_minus = (h_lm_sum + h_lm_diff)*(jnp.abs(h_lm_diff) > 1e-12) 

        return h_lm_plus, h_lm_minus

    @staticmethod
    @partial(jax.vmap, in_axes=(None, None, 1))
    def interp_omega(
        time_grid: Float[Array, " n_grid"],
        time: Float[Array, " n_sample"],
        Omega: Float[Array, " n_Omega"],
    ):
        return CubicSpline(time_grid, Omega)(time)
    
    def multiply_quats(self, q1, q2):
        return jnp.array([
                q1[:,0]*q2[:,0] - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2] - q1[:,3]*q2[:,3],
                q1[:,2]*q2[:,3] - q2[:,2]*q1[:,3] + q1[:,0]*q2[:,1] + q2[:,0]*q1[:,1],
                q1[:,3]*q2[:,1] - q2[:,3]*q1[:,1] + q1[:,0]*q2[:,2] + q2[:,0]*q1[:,2],
                q1[:,1]*q2[:,2] - q2[:,1]*q1[:,2] + q1[:,0]*q2[:,3] + q2[:,0]*q1[:,3]])

    def wigner_d_coefficients(
        self, 
        quat: Float[Array, " n_quat n_sample"],
        orbphase: Float[Array, "n_sample"],
        mode: tuple
        ) -> Float[Array, " n_modes n_sample"]:

        # First rotate the quaternion as well
        quat_rot = jnp.array([jnp.cos(orbphase / 2.), 0. * orbphase, 0. * orbphase, jnp.sin(orbphase / 2.)]).T
        quat_full = self.multiply_quats(quat, quat_rot).T

        # Need to invert the quaternions
        q_conj = -quat_full
        q_conj = q_conj.at[:,0].set(-q_conj[:,0])

        quat_inv = (q_conj.T/self.multiply_quats(quat_full, q_conj)[0]).T

        # Construct the rotations
        R_A = quat_inv[:,0] + 1j * quat_inv[:,3]
        R_B = quat_inv[:,2] + 1j * quat_inv[:,1]

        abs_R_ratio = jnp.abs(R_B)/jnp.abs(R_A)

        R_A_small = (jnp.abs(R_A) < 1e-12)
        R_B_small = (jnp.abs(R_B) < 1e-12)

        i1 = jnp.where((1-R_A_small) * (1-R_B_small))
        i2 = jnp.where(R_A_small)
        i3 = jnp.where((1-R_A_small) * R_B_small)

        matrix_coefs = jnp.zeros((quat_inv.shape[0],  self.n_modes_extended), dtype=complex)
        
        # Handling the if statements, additionally using. a Dirac delta to ensure the ells match
        ell_p, m_p = mode
        for (ell, m), i in self.modelist_dict_extended.items():

            matrix_coefs = matrix_coefs.at[i2, i].set(float(ell_p == ell) * float(m_p == -m) * R_B[i2] ** (2 * m) * (-1) ** (ell + m - 1))
            matrix_coefs = matrix_coefs.at[i3, i].set(float(ell_p == ell) * float(m_p == m) * R_A[i3] ** (2 * m))

            factor = jnp.abs(R_A[i1]) ** (2*ell - 2*m) * R_A[i1] ** (m + m_p)  * R_B[i1] ** (m - m_p) * \
                jnp.sqrt((factorial(ell+m)*(factorial(ell-m)))/(factorial(ell+m_p)*(factorial(ell-m_p))))
            summation = jnp.sum(jnp.array([(-1)**rho * comb(ell + m_p, rho) * comb(ell - m_p, ell - rho - m) * 
                abs_R_ratio[i1]**(2*rho) for rho in range(max(0, m_p - m), min(ell + m_p, ell - m)+1)]),axis=0)
            
            matrix_coefs = matrix_coefs.at[i1, i].set(float(ell_p == ell) * factor * summation)

        # Check the gradient of this (masking out nans)
        matrix_coefs = matrix_coefs.at[jnp.isnan(matrix_coefs)].set(0.0)

        return matrix_coefs

    def get_waveform(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_dim"],
        theta: float = 0.0,
        phi: float = 0.0,
        # quaternions
        init_quat: Float[Array, " n_quat"] = jnp.array([1.0, 0.0, 0.0, 0.0]),
        init_orb_phase: float = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        # TODO set up the appropriate t_low etc

        # Initialize Omega with structure:
        # Omega = [Quaterion, Orb phase, spin_1, spin_2]
        # Note that the spins are in the coprecessing frame
        q = params[0]
        Omega_0 = jnp.concatenate([init_quat, jnp.array([init_orb_phase]), params[1:]])

        normA = jnp.linalg.norm(params[1:4])
        normB = jnp.linalg.norm(params[4:7])

        predictors_parameters, n_max = eqx.partition(self.data.coorb, eqx.is_array)
        dt = self.data.diff_t_ds

        # TODO start construction zone
        # Start the timestepping process
        init_state = (Omega_0, q, normA, normB)

        ############# Hacking predictor
        n_steps = 3
        new_coefs = predictors_parameters.coefs
        new_bfOrders = predictors_parameters.bfOrders

        new_coefs = new_coefs[jnp.array([[0,1,1,2], [2,3,3,4], [4,5,5,6]])]
        new_bfOrders = new_bfOrders[jnp.array([[0,1,1,2], [2,3,3,4], [4,5,5,6]])]
        new_dt = dt[:2*n_steps][jnp.array([[0,1,1,2], [2,3,3,4], [4,5,5,6]])]
        t_ds_rk4 = self.data.t_ds[:2*n_steps:2]

        predictor = eqx.filter_vmap(eqx.filter_vmap(in_axes=(0, 0, None))(make_polypredictor_ensemble), in_axes=(0, 0, None))(new_coefs, new_bfOrders, 100)
        predictor_parameters_new, n_max = eqx.partition(predictor, eqx.is_array)

        # RK4 for n_steps
        ########### End hacking predictor

        # RK4 for 3 steps
        def RK4_kernel(
            carry: tuple[Float[Array, " n_Omega"],  Float, Float, Float], data
        ) -> tuple[
            tuple[Float[Array, " n_Omega"], Float, Float, Float],
            tuple[Float[Array, " n_Omega"], Float[Array, " n_Omega"]]
        ]:
            Omega, q, normA, normB = carry
            predictors_parameters, dt = data
            predictor = eqx.combine(predictors_parameters, n_max)
            Omega_next_unnormalized, k_next = self.RK4(q, Omega, predictor, dt)
            Omega_next = self.normalize_Omega(Omega_next_unnormalized, normA, normB)

            return (Omega_next, q, normA, normB), (Omega_next, k_next)
        

        state, (Omega_rk4, dOmega_dt_rk4) = jax.lax.scan(RK4_kernel, init_state, (predictor_parameters_new, new_dt))

        # Iterating forward to every second step because we need the intermediate steps to evaluate RK4
        # This is a result of the PolyPredictor being defined on a fixed dynamical timescale grid
        # for i, dt in enumerate(self.data.diff_t_ds[:2 * n_steps:2]):
            
        #     # TODO check with Vijay about non-uniform dt
        #     k1 = self.get_Omega_derivative_from_index(Omega_ab4[i], q, predictor, 2*i)
        #     k2 = self.get_Omega_derivative_from_index(Omega_ab4[i] + k1 * dt, q, predictor, 2*i+1)
        #     k3 = self.get_Omega_derivative_from_index(Omega_ab4[i] + k2 * dt, q, predictor, 2*i+1)
        #     k4 = self.get_Omega_derivative_from_index(Omega_ab4[i] + k3 * 2*dt, q, predictor, 2*i+2)

        #     Omega_next = Omega_ab4[i] + (dt/3) * (k1 + 2*k2 + 2*k3 + k4)
            
        #     Omega_ab4 = Omega_ab4.at[i+1].set(self.normalize_Omega(Omega_next, normA, normB)) # TODO change to RK4
        #     k_ab4 = k_ab4.at[i].set(k1)
        #     dt_ab4 = dt_ab4.at[i].set(2*dt)

        # AB4 for N-3 steps
        def timestepping_kernel(
            carry: tuple[Float[Array, " 4 n_Omega"], Float[Array, " 3 n_Omega"], Float, Float, Float], data
        ) -> tuple[
            tuple[Float[Array, " 4 n_Omega"], Float[Array, " 3 n_Omega"], Float, Float, Float],
            Float[Array, " n_Omega"],
        ]:
            Omega, k_ab4, q, normA, normB = carry
            predictors_parameters, dt = data
            predictor = eqx.combine(predictors_parameters, n_max)
            Omega_next_unnormalized, k_next = self.AB4(q, Omega, k_ab4, predictor, dt)
            Omega_next = self.normalize_Omega(Omega_next_unnormalized, normA, normB)

            Omega = jnp.concatenate((Omega[1:], Omega_next[jnp.newaxis,:]),axis=0)
            k_ab4 = jnp.concatenate((k_ab4[1:], k_next[jnp.newaxis,:]),axis=0)
            return (Omega, k_ab4, q, normA, normB), Omega_next

        # integral timestepper
        # scan expect a function and initial stata, plus the data
        # Building the new predictor with the first 6 timesteps cut off
        ab4_coefs = predictors_parameters.coefs
        ab4_bfOrders = predictors_parameters.bfOrders

        ab4_coefs = ab4_coefs[2*n_steps:]
        ab4_bfOrders = ab4_bfOrders[2*n_steps:]

        
        t_ds_ab4 = self.data.t_ds[2*n_steps:]
        t_ds_array = jnp.concatenate([t_ds_rk4, t_ds_ab4])

        dt_combined = jnp.diff(t_ds_array)

        dt_ab4 = dt_combined[jnp.array([[i-3, i-2, i-1, i] for i in range(n_steps, len(dt_combined))])]

        predictor = eqx.filter_vmap(make_polypredictor_ensemble)(ab4_coefs, ab4_bfOrders, 100)
        predictor_parameters_ab4, n_max = eqx.partition(predictor, eqx.is_array)


        init_state_AB4 = (Omega_rk4, dOmega_dt_rk4, q, normA, normB)
        state, Omega = jax.lax.scan(timestepping_kernel, init_state_AB4, (predictor_parameters_ab4, dt_ab4))
        Omega = jnp.concatenate([Omega_0[jnp.newaxis,:], Omega_rk4, Omega], axis=0)

        print(Omega.shape)
        print(t_ds_array.shape)

        # TODO end construction zone 

        # Interpolating to the coorbital time array
        Omega_interp = self.interp_omega(t_ds_array, self.data.t_coorb, Omega).T

        # Normalizing the quaternions after interpolation
        Omega_interp = Omega_interp.at[:,:4].set((Omega_interp[:,:4].T/(jnp.sqrt(jnp.sum(jnp.abs(Omega_interp[:,:4])**2, axis=1)) +1.e-12)).T)

        # Get the lambda parameters to go into the waveform calculation
        lambdas = jax.vmap(self._get_fit_params)(
            jax.vmap(self._get_coorb_params, in_axes=(None, 0))(q, Omega_interp)
        )

        # TODO need to work out how to vmap this later
        inertial_h_lms = jnp.zeros((len(self.data.t_coorb), self.n_modes_extended), dtype=complex)

        for mode in self.modelist_dict.keys():
            # get the coorb hlms
            coorb_h_lm_plus, coorb_h_lm_minus = self.get_coorb_hlm(lambdas, mode=mode)

            # Multiply copressing mode by Wigner-D components (N_modes x times)
            # Note that this also does the rotation of the quaternions into the inertial frame
            inertial_h_lms += (self.wigner_d_coefficients(Omega_interp[:,:4], Omega_interp[:,4], mode).T * coorb_h_lm_plus).T
            inertial_h_lms += (self.wigner_d_coefficients(Omega_interp[:,:4], Omega_interp[:,4], (mode[0], -mode[1])).T * coorb_h_lm_minus).T

        # Sum along the N_modes axis with the spherical harmonics to generate strain as function of time
        inertial_h = jnp.zeros(len(self.data.t_coorb), dtype=complex)
        for idx in self.modelist_dict_extended.values():
            # Note the LAL convention for the phasing
            inertial_h += self.harmonics[idx](theta, jnp.pi/2 - phi) * inertial_h_lms[:,idx]

        return inertial_h, Omega_interp
