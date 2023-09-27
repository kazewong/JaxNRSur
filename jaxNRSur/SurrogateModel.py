import h5py
import jax.numpy as jnp
from jaxNRSur.EIMPredictor import EIMpredictor
from jaxNRSur.spline import CubicSpline
from jaxtyping import Array, Float
from read_node_info import read_node_fit_info_from_h5


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
    def __init__(self, data_path):
        data = h5py.File(data_path, "r")
        self.sur_time = jnp.array(data["domain"])

        self.amp_data = data["sur_subs/ITEM_8/func_subs/ITEM_1/"]
        self.phase_data = data["sur_subs/ITEM_8/func_subs/ITEM_0/"]

        assert isinstance(self.amp_data, h5py.Group), "amp_data is not a group"
        assert isinstance(self.phase_data, h5py.Group), "phase_data is not a group"

        self.amp_B_matrix = jnp.array(self.amp_data["ei_basis"])
        self.amp_gpr_predictors = self.initialize_EIM_predictor(self.amp_data)
        self.n_amp_nodes = len(self.amp_gpr_predictors)

        self.phase_B_matrix = jnp.array(self.phase_data["ei_basis"])
        self.phase_gpr_predictors = self.initialize_EIM_predictor(self.phase_data)
        self.n_phase_nodes = len(self.phase_gpr_predictors)

    def initialize_EIM_predictor(self, node_data):
        try:
            n_nodes = node_data["n_nodes"][()]
        except ValueError:
            raise ValueError("n_nodes data doesn't exist")

        predictors = []
        for count in range(n_nodes):
            try:
                h5_GPR_datadump = node_data[
                    "node_functions/ITEM_%d/node_function/DICT_fit_data" % (count)
                ]
            except ValueError:
                raise ValueError("GPR Fit info doesn't exist")
            res = read_node_fit_info_from_h5(h5_GPR_datadump)
            node_predictor = EIMpredictor(res)
            predictors.append(node_predictor)
        return predictors

    def get_eim(
        self, params: Float[Array, str("n_dim")]
    ) -> tuple[Float[Array, str("n_sample")], Float[Array, str("n_sample")]]:
        amp_result = jnp.zeros(self.n_amp_nodes)
        phase_result = jnp.zeros(self.n_phase_nodes)
        for i in range(self.n_amp_nodes):
            amp_result = amp_result.at[i].set(self.amp_gpr_predictors[i](params)[0])
        for i in range(self.n_phase_nodes):
            phase_result = phase_result.at[i].set(
                self.phase_gpr_predictors[i](params)[0]
            )
        return amp_result, phase_result

    def get_waveform(
        self, time: Float[Array, str("n_sample")], params: dict
    ) -> Float[Array, str("n_sample")]:
        chi_hat, chi_a = effective_spin(params[0], params[1], params[2])
        trans_params = jnp.array([[jnp.log(params[0]), chi_hat, chi_a]])
        amp_result, phase_result = self.get_eim(trans_params)
        amp = jnp.dot(self.amp_B_matrix.T, amp_result)
        phase = -jnp.dot(self.phase_B_matrix.T, phase_result)
        phase = phase + get_T3_phase(params[0], self.sur_time)
        interp_amp = CubicSpline(self.sur_time, amp)(time)
        interp_phase = CubicSpline(self.sur_time, phase)(time)
        return interp_amp * jnp.exp(1j * interp_phase)
