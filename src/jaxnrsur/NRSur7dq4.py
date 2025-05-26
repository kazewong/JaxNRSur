from functools import partial
import jax.numpy as jnp
import jax
from jax.scipy.special import factorial
from jaxnrsur.Spline import CubicSpline
from jaxnrsur.Harmonics import SpinWeightedSphericalHarmonics
from jaxnrsur.DataLoader import load_data, h5Group_to_dict
from jaxnrsur.special_function import comb
from jaxnrsur.PolyPredictor import (
    PolyPredictor,
    evaluate_ensemble,
    evaluate_ensemble_dynamics,
    make_polypredictor_ensemble,
)
from jaxtyping import Array, Float
import equinox as eqx


class NRSur7dq4DataLoader(eqx.Module):
    t_coorb: Float[Array, " n_sample"]
    t_ds: Float[Array, " n_dynam"]
    diff_t_ds: Float[Array, " n_dynam"]

    modes: list[dict]
    rk4_predictor: PolyPredictor
    rk4_dt: Float[Array, " n_rk_steps"]
    ab4_predictor: PolyPredictor
    ab4_dt: Float[Array, " n_ab4_steps"]
    n_max: PolyPredictor
    t_ds_array: Float[Array, " n_samples"]

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
    ) -> None:
        """
        Initialize the data loader for the NRSur7dq4 model

        Args:
            path (str): Path to the HDF5 file
            modelist (list[tuple[int, int]], optional): List of modes to load.
            Defaults to [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1),
                (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)].
        """
        h5_file = load_data(
            "https://zenodo.org/records/3348115/files/NRSur7dq4.h5?download=1",
            "NRSur7dq4.h5",
        )

        data = h5Group_to_dict(h5_file)
        self.t_coorb = jnp.array(data["t_coorb"])
        self.t_ds = jnp.array(data["t_ds"])
        self.diff_t_ds = jnp.diff(self.t_ds)

        coorb_nmax = -100
        basis_nmax = -100
        for key in data:
            if key.startswith("ds_node"):
                for coorb_key in data[key]:
                    coorb_nmax = max(coorb_nmax, data[key][coorb_key].shape[0])
            if key.startswith("hCoorb"):
                for coorb_key in data[key]["nodeModelers"]:
                    basis_nmax = max(
                        basis_nmax, data[key]["nodeModelers"][coorb_key].shape[0]
                    )

        self.modes = []
        for i in range(len(modelist)):
            self.modes.append(
                self.read_single_mode(data, modelist[i], n_max=basis_nmax)
            )

        assert coorb_nmax == basis_nmax, "coorb_nmax must equal to basis_nmax"

        # TODO: Remove the following self.coorb poly predictor
        coorb = self.read_coorb(data, coorb_nmax)
        predictors_parameters, n_max = eqx.partition(coorb, eqx.is_array)
        self.n_max = n_max

        # TODO: Initialize rk4_polypredictor

        n_steps = 3
        rk4_coefs = predictors_parameters.coefs
        rk4_bfOrders = predictors_parameters.bfOrders

        rk4_coefs = rk4_coefs[jnp.array([[0, 1, 1, 2], [2, 3, 3, 4], [4, 5, 5, 6]])]
        rk4_bfOrders = rk4_bfOrders[
            jnp.array([[0, 1, 1, 2], [2, 3, 3, 4], [4, 5, 5, 6]])
        ]
        self.rk4_dt = self.diff_t_ds[: 2 * n_steps][
            jnp.array([[0, 1, 1, 2], [2, 3, 3, 4], [4, 5, 5, 6]])
        ]
        t_ds_rk4 = self.t_ds[: 2 * n_steps : 2]

        predictor = eqx.filter_vmap(
            eqx.filter_vmap(in_axes=(0, 0, None))(make_polypredictor_ensemble),
            in_axes=(0, 0, None),
        )(rk4_coefs, rk4_bfOrders, coorb_nmax)
        predictor_parameters_new, n_max = eqx.partition(predictor, eqx.is_array)
        self.rk4_predictor = predictor_parameters_new

        # TODO: Initialize ab4_polypredictor

        # integral timestepper
        # scan expect a function and initial stata, plus the data
        # Building the new predictor with the first 6 timesteps cut off
        ab4_coefs = predictors_parameters.coefs
        ab4_bfOrders = predictors_parameters.bfOrders

        ab4_coefs = ab4_coefs[2 * n_steps :]
        ab4_bfOrders = ab4_bfOrders[2 * n_steps :]

        t_ds_ab4 = self.t_ds[2 * n_steps :]
        self.t_ds_array = jnp.concatenate([t_ds_rk4, t_ds_ab4])

        dt_combined = jnp.diff(self.t_ds_array)

        self.ab4_dt = dt_combined[
            jnp.array(
                [[i - 3, i - 2, i - 1, i] for i in range(n_steps, len(dt_combined))]
            )
        ]

        predictor = eqx.filter_vmap(make_polypredictor_ensemble)(
            ab4_coefs, ab4_bfOrders, 100
        )
        predictor_parameters_ab4, n_max = eqx.partition(predictor, eqx.is_array)
        self.ab4_predictor = predictor_parameters_ab4

    def read_mode_function(self, node_data: dict, n_max: int) -> dict:
        result = {}
        n_nodes = len(node_data["nodeIndices"])  # type: ignore
        result["n_nodes"] = n_nodes

        coefs = []
        bfOrders = []

        for count in range(n_nodes):  # n_nodes is the n which you iterate over
            coef = node_data["nodeModelers"][f"coefs_{count}"]
            bfOrder = node_data["nodeModelers"][f"bfOrders_{count}"]

            coefs.append(jnp.pad(coef, (0, n_max - len(coef))))
            bfOrders.append(jnp.pad(bfOrder, ((0, n_max - len(bfOrder)), (0, 0))))

        result["predictors"] = make_polypredictor_ensemble(
            jnp.array(coefs), jnp.array(bfOrders), n_max
        )
        result["eim_basis"] = jnp.array(node_data["EIBasis"])
        result["node_indices"] = jnp.array(node_data["nodeIndices"])
        return result

    def read_single_mode(self, file: dict, mode: tuple[int, int], n_max: int) -> dict:
        result = {}
        if mode[1] > 0:
            result["real_plus"] = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_Re+"], n_max
            )
            result["imag_plus"] = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_Im+"], n_max
            )
            result["real_minus"] = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_Re-"], n_max
            )
            result["imag_minus"] = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_Im-"], n_max
            )
        else:
            result["real_plus"] = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_real"], n_max
            )
            # result['real_minus'] = 0
            # TODO Make the structure of the m=0 modes similar
            # to hangle in the same way as m != 0

            result["imag_plus"] = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_imag"], n_max
            )

            node_data = {
                "nodeModelers": {
                    "coefs_0": jnp.array([0]),
                    "bfOrders_0": jnp.zeros((0, 7)),
                },
                "nodeIndices": jnp.array([0]),
                "EIBasis": jnp.array([0]),
            }
            result["real_minus"] = self.read_mode_function(node_data, 1)
            result["imag_minus"] = self.read_mode_function(node_data, 1)

        result["mode"] = mode

        return result

    def read_coorb(self, file: dict, n_max: int) -> PolyPredictor:
        result = []

        tags = [
            "chiA_0",
            "chiA_1",
            "chiA_2",
            "chiB_0",
            "chiB_1",
            "chiB_2",
            "omega",
            "omega_orb_0",
            "omega_orb_1",
        ]

        @eqx.filter_vmap(in_axes=(0, 0))
        def combine_poly_predictors(
            coefs: Float[Array, " n_coefs n_order"], bfOrders: Float[Array, " n_order"]
        ) -> PolyPredictor:
            return make_polypredictor_ensemble(coefs, bfOrders, n_max)

        coefs = []
        bfOrders = []

        for i in range(len(self.t_ds) - 1):
            local_coefs = []
            local_bfOrders = []

            for tag in tags:
                coef = file[f"ds_node_{i}"][f"{tag}_coefs"]
                bfOrder = file[f"ds_node_{i}"][f"{tag}_bfOrders"]
                local_coefs.append(jnp.pad(coef, (0, n_max - len(coef))))
                local_bfOrders.append(
                    jnp.pad(bfOrder, ((0, n_max - len(bfOrder)), (0, 0)))
                )

            coefs.append(jnp.stack(local_coefs))
            bfOrders.append(jnp.stack(local_bfOrders))

        coefs = jnp.stack(coefs)
        bfOrders = jnp.stack(bfOrders)
        result = combine_poly_predictors(coefs, bfOrders)

        return result


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

        fit_params = fit_params.at[0].set(q_fit_offset + q_fit_slope * jnp.log(q))
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
        ) = evaluate_ensemble_dynamics(predictor, fit_params)

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

    def AB4(
        self,
        q: Float,
        Omega_i4: Float[Array, " 4 n_Omega"],
        k_ab4: Float[Array, " 3 n_Omega"],
        predictor,
        dt: Float[Array, " 4"],
    ) -> tuple[Float[Array, " n_Omega"], Float[Array, " n_Omega"]]:

        dOmega_dt = self.get_Omega_derivative(Omega_i4[-1], q, predictor)

        # Using the timestep variable AB4 from gwsurrogate
        dt1 = dt[0]
        dt2 = dt[1]
        dt3 = dt[2]
        dt4 = dt[3]

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

        A = dOmega_dt
        B = dOmega_dt * B4 - k_ab4[0] * B41 - k_ab4[1] * B42 - k_ab4[2] * B43
        C = dOmega_dt * C4 - k_ab4[0] * C41 - k_ab4[1] * C42 - k_ab4[2] * C43
        D = (
            (dOmega_dt - k_ab4[0]) / D1
            - (dOmega_dt - k_ab4[1]) / D2
            + (dOmega_dt - k_ab4[2]) / D3
        )

        return (
            Omega_i4[-1]
            + dt4 * (A + dt4 * (0.5 * B + dt4 * (C / 3.0 + dt4 * 0.25 * D))),
            dOmega_dt,
        )

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
            data: tuple[PolyPredictor, Float[Array, " 4"]],
        ):

            Omega, q, derivative = carry
            predictor_parameters, dt = data

            predictors = eqx.combine(predictor_parameters, n_max)
            derivative = self.get_Omega_derivative(
                Omega + dt * derivative, q, predictors
            )

            return (Omega, q, derivative), derivative

        state, dOmega_dt_rk4 = jax.lax.scan(
            get_RK4_Omega_derivatives,
            (Omega, q, jnp.zeros(len(Omega))),
            (predictor_parameters, jnp.array([0, 1, 1, 2]) * dt),
        )

        Omega_next = Omega + (1.0 / 3.0) * (
            dt[0] * dOmega_dt_rk4[0]
            + 2 * dt[0] * dOmega_dt_rk4[1]
            + 2 * dt[0] * dOmega_dt_rk4[2]
            + dt[0] * dOmega_dt_rk4[3]
        )
        predictor_i = make_polypredictor_ensemble(
            predictors.coefs[0], predictors.bfOrders[0], 100
        )
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
        h_lm_plus = jnp.conj(h_lm_sum - h_lm_diff) * (
            jnp.abs(h_lm_diff) > 1e-12
        ) + h_lm_sum * (jnp.abs(h_lm_diff) < 1e-12)
        h_lm_minus = (h_lm_sum + h_lm_diff) * (jnp.abs(h_lm_diff) > 1e-12)

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
        return jnp.array(
            [
                q1[:, 0] * q2[:, 0]
                - q1[:, 1] * q2[:, 1]
                - q1[:, 2] * q2[:, 2]
                - q1[:, 3] * q2[:, 3],
                q1[:, 2] * q2[:, 3]
                - q2[:, 2] * q1[:, 3]
                + q1[:, 0] * q2[:, 1]
                + q2[:, 0] * q1[:, 1],
                q1[:, 3] * q2[:, 1]
                - q2[:, 3] * q1[:, 1]
                + q1[:, 0] * q2[:, 2]
                + q2[:, 0] * q1[:, 2],
                q1[:, 1] * q2[:, 2]
                - q2[:, 1] * q1[:, 2]
                + q1[:, 0] * q2[:, 3]
                + q2[:, 0] * q1[:, 3],
            ]
        )

    def wigner_d_coefficients(
        self,
        quat: Float[Array, " n_quat n_sample"],
        orbphase: Float[Array, " n_sample"],
        mode: tuple,
    ) -> Float[Array, " n_modes n_sample"]:

        # First rotate the quaternion as well
        quat_rot = jnp.array(
            [
                jnp.cos(orbphase / 2.0),
                0.0 * orbphase,
                0.0 * orbphase,
                jnp.sin(orbphase / 2.0),
            ]
        ).T
        quat_full = self.multiply_quats(quat, quat_rot).T

        # Need to invert the quaternions
        q_conj = -quat_full
        q_conj = q_conj.at[:, 0].set(-q_conj[:, 0])

        quat_inv = (q_conj.T / self.multiply_quats(quat_full, q_conj)[0]).T

        # Construct the rotations
        R_A = quat_inv[:, 0] + 1j * quat_inv[:, 3]
        R_B = quat_inv[:, 2] + 1j * quat_inv[:, 1]

        abs_R_ratio = jnp.abs(R_B) / jnp.abs(R_A)

        R_A_small = jnp.abs(R_A) < 1e-12
        R_B_small = jnp.abs(R_B) < 1e-12

        i1 = jnp.where((1 - R_A_small) * (1 - R_B_small))
        i2 = jnp.where(R_A_small)
        i3 = jnp.where((1 - R_A_small) * R_B_small)

        matrix_coefs = jnp.zeros(
            (quat_inv.shape[0], self.n_modes_extended), dtype=complex
        )

        # Handling the if statements, additionally using. a Dirac delta to ensure the ells match
        ell_p, m_p = mode
        for (ell, m), i in self.modelist_dict_extended.items():

            matrix_coefs = matrix_coefs.at[i2, i].set(
                float(ell_p == ell)
                * float(m_p == -m)
                * R_B[i2] ** (2 * m)
                * (-1) ** (ell + m - 1)
            )
            matrix_coefs = matrix_coefs.at[i3, i].set(
                float(ell_p == ell) * float(m_p == m) * R_A[i3] ** (2 * m)
            )

            factor = (
                jnp.abs(R_A[i1]) ** (2 * ell - 2 * m)
                * R_A[i1] ** (m + m_p)
                * R_B[i1] ** (m - m_p)
                * jnp.sqrt(
                    (factorial(ell + m) * (factorial(ell - m)))
                    / (factorial(ell + m_p) * (factorial(ell - m_p)))
                )
            )
            summation = jnp.sum(
                jnp.array(
                    [
                        (-1) ** rho
                        * comb(ell + m_p, rho)
                        * comb(ell - m_p, ell - rho - m)
                        * abs_R_ratio[i1] ** (2 * rho)
                        for rho in range(max(0, m_p - m), min(ell + m_p, ell - m) + 1)
                    ]
                ),
                axis=0,
            )

            matrix_coefs = matrix_coefs.at[i1, i].set(
                float(ell_p == ell) * factor * summation
            )

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

        dt = self.data.diff_t_ds

        # TODO start construction zone
        # Start the timestepping process
        init_state = (Omega_0, q, normA, normB)

        # RK4 for 3 steps
        def RK4_kernel(
            carry: tuple[Float[Array, " n_Omega"], Float, Float, Float], data
        ) -> tuple[
            tuple[Float[Array, " n_Omega"], Float, Float, Float],
            tuple[Float[Array, " n_Omega"], Float[Array, " n_Omega"]],
        ]:
            Omega, q, normA, normB = carry
            predictors_parameters, dt = data
            predictor = eqx.combine(predictors_parameters, self.data.n_max)
            Omega_next_unnormalized, k_next = self.RK4(q, Omega, predictor, dt)
            Omega_next = self.normalize_Omega(Omega_next_unnormalized, normA, normB)

            return (Omega_next, q, normA, normB), (Omega_next, k_next)

        state, (Omega_rk4, dOmega_dt_rk4) = jax.lax.scan(
            RK4_kernel, init_state, (self.data.rk4_predictor, self.data.rk4_dt)
        )

        # AB4 for N-3 steps
        def timestepping_kernel(
            carry: tuple[
                Float[Array, " 4 n_Omega"],
                Float[Array, " 3 n_Omega"],
                Float,
                Float,
                Float,
            ],
            data: tuple[PolyPredictor, Float[Array, " n_samples"]],
        ) -> tuple[
            tuple[
                Float[Array, " 4 n_Omega"],
                Float[Array, " 3 n_Omega"],
                Float,
                Float,
                Float,
            ],
            Float[Array, " n_Omega"],
        ]:
            Omega, k_ab4, q, normA, normB = carry
            predictors_parameters, dt = data
            predictor = eqx.combine(predictors_parameters, self.data.n_max)
            Omega_next_unnormalized, k_next = self.AB4(q, Omega, k_ab4, predictor, dt)
            Omega_next = self.normalize_Omega(Omega_next_unnormalized, normA, normB)

            Omega = jnp.concatenate((Omega[1:], Omega_next[jnp.newaxis, :]), axis=0)
            k_ab4 = jnp.concatenate((k_ab4[1:], k_next[jnp.newaxis, :]), axis=0)
            return (Omega, k_ab4, q, normA, normB), Omega_next

        init_state_AB4 = (Omega_rk4, dOmega_dt_rk4, q, normA, normB)
        state, Omega = jax.lax.scan(
            timestepping_kernel,
            init_state_AB4,
            (self.data.ab4_predictor, self.data.ab4_dt),
        )
        Omega = jnp.concatenate([Omega_0[jnp.newaxis, :], Omega_rk4, Omega], axis=0)

        # TODO end construction zone

        # Interpolating to the coorbital time array
        Omega_interp = self.interp_omega(
            self.data.t_ds_array, self.data.t_coorb, Omega
        ).T

        # Normalizing the quaternions after interpolation
        Omega_interp = Omega_interp.at[:, :4].set(
            (
                Omega_interp[:, :4].T
                / (
                    jnp.sqrt(jnp.sum(jnp.abs(Omega_interp[:, :4]) ** 2, axis=1))
                    + 1.0e-12
                )
            ).T
        )

        # Get the lambda parameters to go into the waveform calculation
        lambdas = jax.vmap(self._get_fit_params)(
            jax.vmap(self._get_coorb_params, in_axes=(None, 0))(q, Omega_interp)
        )

        # TODO need to work out how to vmap this later
        inertial_h_lms = jnp.zeros(
            (len(self.data.t_coorb), self.n_modes_extended), dtype=complex
        )

        for mode in self.modelist_dict.keys():
            # get the coorb hlms
            coorb_h_lm_plus, coorb_h_lm_minus = self.get_coorb_hlm(lambdas, mode=mode)

            # Multiply copressing mode by Wigner-D components (N_modes x times)
            # Note that this also does the rotation of the quaternions into the inertial frame
            inertial_h_lms += (
                self.wigner_d_coefficients(
                    Omega_interp[:, :4], Omega_interp[:, 4], mode
                ).T
                * coorb_h_lm_plus
            ).T
            inertial_h_lms += (
                self.wigner_d_coefficients(
                    Omega_interp[:, :4], Omega_interp[:, 4], (mode[0], -mode[1])
                ).T
                * coorb_h_lm_minus
            ).T

        # Sum along the N_modes axis with the spherical harmonics to generate strain as function of time
        inertial_h = jnp.zeros(len(self.data.t_coorb), dtype=complex)
        for idx in self.modelist_dict_extended.values():
            # Note the LAL convention for the phasing
            inertial_h += (
                self.harmonics[idx](theta, jnp.pi / 2 - phi) * inertial_h_lms[:, idx]
            )

        # TODO: Add interpolation on time grid

        return inertial_h, Omega_interp
