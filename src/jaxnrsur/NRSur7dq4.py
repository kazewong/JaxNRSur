from functools import partial
from jax.lax import abs_p
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
    stable_power,
)
from jaxtyping import Array, Float
import equinox as eqx

class NRSur7dq4ModeFunction(eqx.Module):
    predictors: PolyPredictor
    eim_basis: Float[Array, " n_nodes n_sample"]
    node_indices: Float[Array, " n_nodes"]

    @property
    def n_nodes(self) -> int:
        return len(self.node_indices)

    def __init__(
        self,
        predictors: PolyPredictor,
        eim_basis: Float[Array, " n_nodes n_sample"],
        node_indices: Float[Array, " n_nodes"],
    ) -> None:
        """
        Initialize the mode functions with predictors, EIM basis, and node indices.
        Args:
            predictors (PolyPredictor): The polynomial predictors for the mode.
            eim_basis (Float[Array, " n_nodes n_sample"]): The EIM basis for the mode.
            node_indices (Float[Array, " n_nodes"]): The indices of the nodes in the EIM basis.
        """
        self.predictors = predictors
        self.eim_basis = eim_basis
        self.node_indices = node_indices

class NRSur7dq4Mode:
    real_plus: NRSur7dq4ModeFunction
    imag_plus: NRSur7dq4ModeFunction
    real_minus: NRSur7dq4ModeFunction
    imag_minus: NRSur7dq4ModeFunction
    mode: tuple[int, int]

    def __init__(
        self,
        real_plus: NRSur7dq4ModeFunction,
        imag_plus: NRSur7dq4ModeFunction,
        real_minus: NRSur7dq4ModeFunction,
        imag_minus: NRSur7dq4ModeFunction,
        mode: tuple[int, int],
    ) -> None:
        """
        Initialize the modes with their respective functions and mode tuple.

        Args:
            real_plus (NRSur7dq4ModeFunctions): The real part of the plus mode.
            imag_plus (NRSur7dq4ModeFunctions): The imaginary part of the plus mode.
            real_minus (NRSur7dq4ModeFunctions): The real part of the minus mode.
            imag_minus (NRSur7dq4ModeFunctions): The imaginary part of the minus mode.
            mode (tuple[int, int]): The mode tuple (l, m).
        """
        self.real_plus = real_plus
        self.imag_plus = imag_plus
        self.real_minus = real_minus
        self.imag_minus = imag_minus
        self.mode = mode


class NRSur7dq4DataLoader(eqx.Module):
    t_coorb: Float[Array, " n_sample"]
    t_ds: Float[Array, " n_dynam"]
    diff_t_ds: Float[Array, " n_dynam"]

    modes: list[NRSur7dq4Mode]
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

    def read_mode_function(self, node_data: dict, n_max: int) -> NRSur7dq4ModeFunction:
        n_nodes = len(node_data["nodeIndices"])

        coefs = []
        bfOrders = []

        for count in range(n_nodes):  # n_nodes is the n which you iterate over
            coef = node_data["nodeModelers"][f"coefs_{count}"]
            bfOrder = node_data["nodeModelers"][f"bfOrders_{count}"]

            coefs.append(jnp.pad(coef, (0, n_max - len(coef))))
            bfOrders.append(jnp.pad(bfOrder, ((0, n_max - len(bfOrder)), (0, 0))))

        predictors = make_polypredictor_ensemble(
            jnp.array(coefs), jnp.array(bfOrders), n_max
        )
        eim_basis = jnp.array(node_data["EIBasis"])
        node_indices = jnp.array(node_data["nodeIndices"])
        return NRSur7dq4ModeFunction(
            predictors=predictors,
            eim_basis=eim_basis,
            node_indices=node_indices,
        )

    def read_single_mode(self, file: dict, mode: tuple[int, int], n_max: int) -> NRSur7dq4Mode:
        if mode[1] > 0:
            real_plus = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_Re+"], n_max
            )
            imag_plus = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_Im+"], n_max
            )
            real_minus = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_Re-"], n_max
            )
            imag_minus = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_Im-"], n_max
            )
        else:
            real_plus = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_real"], n_max
            )
            # result['real_minus'] = 0
            # TODO Make the structure of the m=0 modes similar
            # to hangle in the same way as m != 0

            imag_plus = self.read_mode_function(
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
            real_minus = self.read_mode_function(node_data, 1)
            imag_minus = self.read_mode_function(node_data, 1)

        return NRSur7dq4Mode(
            real_plus=real_plus,
            imag_plus=imag_plus,
            real_minus=real_minus,
            imag_minus=imag_minus,
            mode=mode,
        )

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
    max_lm: tuple[int,int]

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
            self.modelist_dict[i] = mode

        self.modelist_dict_extended = {}
        idx = 0
        for i, mode in enumerate(modelist):
            self.modelist_dict_extended[idx] = mode
            idx += 1

            if mode[1] > 0:
                negative_mode = (mode[0], -mode[1])
                self.modelist_dict_extended[idx] = negative_mode
                idx += 1

        self.n_modes_extended = len(self.modelist_dict_extended.keys())

        for mode in list(self.modelist_dict_extended.values()):
            self.harmonics.append(SpinWeightedSphericalHarmonics(-2, mode[0], mode[1]))
            
        self.max_lm = (max([mode[0] for mode in modelist]), max([abs(mode[1]) for mode in modelist]))

    def __call__(self,
      time: Float[Array, " n_sample"],
      params: Float[Array, " n_dim"],
      theta: float = 0.0,
      phi: float = 0.0,
      # quaternions
      init_quat: Float[Array, " n_quat"] = jnp.array([1.0, 0.0, 0.0, 0.0]),
      init_orb_phase: float = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
      return self.get_waveform_geometric(time,params, theta, phi, init_quat,
                                         init_orb_phase)

    def _get_coorb_params(
        self, q: Float, Omega: Float[Array, " n_Omega"]
    ) -> Float[Array, " n_dim"]:
        # First construct array for coorbital frame
        # borrowing notation from gwsurrogate

        sp = jnp.sin(Omega[4])
        cp = jnp.cos(Omega[4])

        coorb_x = jnp.array(
            [
                q,
                Omega[5] * cp + Omega[6] * sp,
                -Omega[5] * sp + Omega[6] * cp,
                Omega[7],
                Omega[8] * cp + Omega[9] * sp,
                -Omega[8] * sp + Omega[9] * cp,
                Omega[10],
            ]
        )

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

    def get_coorb_hlm(self, lambdas, idx: int):

        # surrogate is built on the symmetric (sum) and antisymmetric (diff)
        # combinations of the +|m| and -|m| modes
        # (although they are confusingly labeled "plus" and "minus" in
        # the file)

        mode = self.data.modes[idx]

        h_lm_sum = self.construct_hlm_from_bases(
            lambdas[mode.real_plus.node_indices],
            mode.real_plus.predictors,
            mode.real_plus.eim_basis,
        ) + 1j * self.construct_hlm_from_bases(
            lambdas[mode.imag_plus.node_indices],
            mode.imag_plus.predictors,
            mode.imag_plus.eim_basis,
        )

        h_lm_diff = self.construct_hlm_from_bases(
            lambdas[mode.real_minus.node_indices],
            mode.real_minus.predictors,
            mode.real_minus.eim_basis,
        ) + 1j * self.construct_hlm_from_bases(
            lambdas[mode.imag_minus.node_indices],
            mode.imag_minus.predictors,
            mode.imag_minus.eim_basis,
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
    @partial(jax.vmap, in_axes=(None, 1, None))
    def interp_omega(
        time_grid: Float[Array, " n_grid"],
        Omega: Float[Array, " n_grid n_omega"],
        time_interp: Float[Array, " n_sample"],
    ) -> Float[Array, " n_sample n_omega"]:
        return CubicSpline(time_grid, Omega)(time_interp)

    @staticmethod
    def multiply_quats(q1, q2):
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
                jnp.zeros_like(orbphase),
                jnp.zeros_like(orbphase),
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

        # The if condition are turned in a lot of vmapped. We should investigate
        # the performance implication
        i1 = (1 - R_A_small) * (1 - R_B_small)
        i2 = (1 - R_B_small) * R_A_small
        i3 = (1 - R_A_small) * R_B_small

        matrix_coefs = jnp.zeros(
            (quat_inv.shape[0], self.n_modes_extended), dtype=complex
        )

        # Handling the if statements, additionally using. a Dirac delta to ensure the ells match
        ell_p, m_p = mode
        
        def wigner_d_kernel(ell, m):
            result = jnp.zeros(quat_inv.shape[0], dtype=complex)
            R_A_prime = jnp.where(
                R_A_small, jnp.zeros_like(R_A), R_A
            )
            R_B_prime = jnp.where(
                R_B_small, jnp.zeros_like(R_B), R_B
            )
            result = jax.lax.select(
                i2 * (ell_p == ell) * (m_p == -m),
                R_B_prime ** (2. * m) * -1 ** (ell + m - 1),
                result,
            )
            result = jax.lax.select(
                i3 * (ell_p == ell) * (m_p == m),
                R_A_prime ** (2 * m),
                result,
            )
            factorial_num = (factorial(ell + m) * (factorial(ell - m)))
            factorial_denom = (factorial(ell + m_p) * (factorial(ell - m_p)))
            factorial_num = jnp.where(factorial_denom == 0, jnp.zeros_like(factorial_num), factorial_num)
            factorial_denom = jnp.where(factorial_denom == 0, jnp.ones_like(factorial_denom), factorial_denom)
            factorial_term = jnp.sqrt(factorial_num / factorial_denom)
            factorial_term = jax.lax.select(
                jnp.isnan(factorial_term), jnp.zeros_like(factorial_term), factorial_term
            )
            term1 = jnp.abs(R_A_prime) ** (2 * ell - 2 * m) * R_A_prime ** (m + m_p)
            term2 = R_B_prime ** (m - m_p)
            term1 = jax.lax.select(~jnp.isfinite(term1), jnp.zeros_like(term1), term1)
            term2 = jax.lax.select(~jnp.isfinite(term2), jnp.zeros_like(term2), term2)

            jax.debug.print("term1: {}, term2: {}", term1, term2)
            jax.debug.print("multiply: {}", term1 * term2)
            factor = jnp.where(
                i1,
                term1 * term2 *
                factorial_term,
                jnp.zeros(R_A_prime.shape).astype(jnp.complexfloating),
            )

            rho = jnp.arange(0, self.max_lm[0] + self.max_lm[1] + 1)
            comb_vmap = jax.vmap(comb, in_axes=(None, 0))
            summation = (((-1) ** rho * comb_vmap(ell + m_p, rho) * comb_vmap(ell - m_p, ell - rho - m)) * abs_R_ratio[:, None] ** (2 * rho)).T
            conditions = (rho >= jnp.array([0, m_p - m]).max()) * (rho <= jnp.array([ell + m_p, ell - m]).min())
            summation = conditions[:, None] * summation
            summation = jnp.sum(summation, axis=0)

            result = jnp.where(
                i1, (ell_p == ell).astype(float) * factor * summation, result
            )
            return result

        # TODO: Find a more jax way to extract the modes
        ells = jnp.array([x[0] for x in self.modelist_dict_extended.values()])
        ms = jnp.array([x[1] for x in self.modelist_dict_extended.values()])

        matrix_coefs = jax.vmap(wigner_d_kernel)(ells, ms).T

        # Check the gradient of this (masking out nans)
        matrix_coefs = jnp.where(
            jnp.isnan(matrix_coefs),
            jnp.zeros(matrix_coefs.shape),
            matrix_coefs,
        )
        
        return matrix_coefs

    def mode_projection(
        self,
        hlm_plus: Float[Array, " n_sample"],
        hlm_minus: Float[Array, " n_sample"],
        quat: Float[Array, " n_quat n_sample"],
        orbphase: Float[Array, " n_sample"],
        mode: tuple[int, int],
    ) -> Float[Array, "n_modes n_sample"]:
        # Get the Wigner D coefficients
        return (self.wigner_d_coefficients(quat, orbphase, mode).T * hlm_plus).T + (self.wigner_d_coefficients(quat, orbphase, (mode[0], -mode[1])).T * hlm_minus).T

    def get_waveform_geometric(
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
        def AB4_kernel(
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
            AB4_kernel,
            init_state_AB4,
            (self.data.ab4_predictor, self.data.ab4_dt),
        )
        # The shape of Omega should be (n_timestep, n_omega)
        Omega = jnp.concatenate([Omega_0[jnp.newaxis, :], Omega_rk4, Omega], axis=0)

        # Interpolating to the coorbital time array
        Omega_interp = self.interp_omega(
            self.data.t_ds_array,
            Omega,
            self.data.t_coorb,
        ).T
        
        Omega_interp = Omega_interp.at[:, :4].set(
            (Omega_interp[:, :4].T /(jnp.sqrt(jnp.sum(Omega_interp[:, :4] ** 2, axis=1)))).T)

        # Get the lambda parameters to go into the waveform calculation
        lambdas = jax.vmap(self._get_fit_params)(
            jax.vmap(self._get_coorb_params, in_axes=(None, 0))(q, Omega_interp)
        )
        
        # TODO need to work out how to vmap this later
        inertial_h_lms = jnp.zeros(
            (len(self.data.t_coorb), self.n_modes_extended), dtype=complex
        )

        # Due to the varying number of nodes, there is no trivial way to vmap this
        coorb_hlm = jnp.array(jax.tree.map(
            lambda idx: self.get_coorb_hlm(lambdas, idx), list(self.modelist_dict.keys()))
        )
        
        hlm_projed = eqx.filter_vmap(
            self.mode_projection,
            in_axes=(0, 0, None, None, 0),
        )(
            coorb_hlm[:, 0],
            coorb_hlm[:, 1],
            Omega_interp[:, :4],
            Omega_interp[:, 4],
            jnp.array(list(self.modelist_dict.values())),
        ).T

        inertial_h_lms += jnp.sum(hlm_projed, axis=-1).T
        
        # Sum along the N_modes axis with the spherical harmonics to generate strain as function of time
        inertial_h = jnp.zeros(len(self.data.t_coorb), dtype=complex)
        for idx in self.modelist_dict_extended.keys():
            # Note the LAL convention for the phasing
            inertial_h += (
                self.harmonics[idx](theta, jnp.pi / 2 - phi) * inertial_h_lms[:, idx]
            )

        # # window surrogate start with a window that is 0 at the start, as well as zero
        # # first and second derivative at the start, and is 1 and zero derivatives
        # # at the end, i.e., x^3(10 + x(6x - 15))
        # t = self.data.t_coorb - self.data.t_coorb[0]
        # # TODO: move this setting somewhere else
        # ALPHA_WINDOW = 0.1
        # x = t / ALPHA_WINDOW / t[-1]
        # window = jnp.where(x < 1, x*x*x*(10 + x*(6*x - 15)), 1.0)
        
        h_re = CubicSpline(self.data.t_coorb, inertial_h.real)(time)
        h_im = CubicSpline(self.data.t_coorb, inertial_h.imag)(time)

        mask = (time >= self.data.t_coorb[0]) * (time <= self.data.t_coorb[-1])
        hp = jnp.where(mask, h_re, 0.)
        hc = jnp.where(mask, -h_im, 0.)
        
        return hp, hc

    def get_waveform_td(self,
                        time: Float[Array, " n_sample"],
                        params: Float[Array, " n_param"],
                        alpha_window: float = 0.0,
                        ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """
        Get the waveform in the time domain in SI units.
        """
        # get scaling parameters
        mtot = params[0]
        dist_mpc = params[1]

        # geometric units to SI
        GMSUN_SI = 1.32712442099000e+20
        C_SI = 2.99792458000000e+08
        RSUN_SI = GMSUN_SI / C_SI**2

        # parsecs to SI
        PC_SI = 3.08567758149136720000e+16
        MPC_SI = 1E6*PC_SI
        
        # form time array with desired sampling rate and duration
        # N = int(seglen*srate)
        # time = jnp.arange(N)/srate - seglen + 2
        
        # evaluate the surrogate over the equivalent geometric time
        time_m = time * C_SI / RSUN_SI / mtot
        hrM_p, hrM_c = self.get_waveform_geometric(time_m,
                                                   jnp.array(params[2:]))

        if alpha_window > 0:
            # create a window for the waveform: the form of the window
            # is chosen such that it is 0 at the start, as well as zero
            # first and second derivative at the start, and is 1 and zero
            # derivatives at the end.
            Tcoorb = self.data.t_coorb[-1] - self.data.t_coorb[0]

            window_start = jnp.max(jnp.array([time_m[0], self.data.t_coorb[0]]))
            window_end = window_start + alpha_window*Tcoorb

            x = (time_m - window_start) / (window_end - window_start)

            window = jnp.select([time_m < window_start, time_m > window_end], 
                                [0.0, 1.0], default=x*x*x*(10 + x*(6*x - 15)))
            hrM_p *= window
            hrM_c *= window
        
        # this is h * r / M, so scale by the mass and distance
        const = mtot * RSUN_SI / dist_mpc / MPC_SI
        return hrM_p * const, hrM_c * const

    def get_waveform_fd(self,
                        time: Float[Array, " n_sample"],
                        params: Float[Array, " n_param"],
                        alpha_window: float = 0.1,
                        ) -> tuple[Float[Array, " n_freq"],
                                   Float[Array, " n_freq"]]:
        """
        Get the waveform in the frequency domain.
        """
        # this could be moved inside get_waveform_td if
        # we change the API to have seglen and delta_t as
        # input, rather than time
        # N = int(seglen/delta_t)
        # time = jnp.arange(N)*delta_t - seglen + 2

        hp_td, hc_td = self.get_waveform_td(time, params,
                                            alpha_window=alpha_window)

        h_fd = jnp.fft.fft(hp_td - 1j*hc_td)
        #f = jnp.fft.fftfreq(N, delta_t)

        # obtain hp_fd and hc_fd
        # rolling the arrays to get the positive and negative frequency components
        # aligned correctly, as in np.fft.rfft
        n = len(h_fd)//2 + 1
        h_fd_positive = h_fd[:n]
        conj_h_fd_negative = jnp.conj(jnp.fft.ifftshift(h_fd))[:n][::-1]
        hp_fd = (h_fd_positive + conj_h_fd_negative)/2
        hc_fd = 1j*(h_fd_positive - conj_h_fd_negative)/2
        return hp_fd, hc_fd