import h5py
from jaxNRSur.EIMPredictor import EIMpredictor
from jaxNRSur.PolyPredictor import PolyPredictor, make_polypredictor_ensemble
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

h5_mode_tuple: dict[tuple[int, int], str] = {
    (2, 0): "ITEM_6",  # No Imaginary part
    (2, 1): "ITEM_5",
    (2, 2): "ITEM_8",
    (3, 0): "ITEM_3",  # No Real part
    (3, 1): "ITEM_4",
    (3, 2): "ITEM_0",
    (3, 3): "ITEM_2",
    (4, 2): "ITEM_9",
    (4, 3): "ITEM_7",
    (4, 4): "ITEM_1",
    (5, 5): "ITEM_10",
}


def h5Group_to_dict(h5_group: h5py.Group) -> dict:
    result = {}
    for key in h5_group.keys():
        local_data = h5_group[key]
        if isinstance(local_data, h5py.Dataset):
            result[key] = local_data[()]
        elif isinstance(local_data, h5py.Group):
            result[key] = h5Group_to_dict(local_data)
        else:
            raise ValueError("Unknown data type")
    return result


class NRHybSur3dq8DataLoader(eqx.Module):
    sur_time: Float[Array, " n_sample"]
    modes: list[dict]

    def __init__(
        self,
        path: str,
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
    ) -> None:
        data = h5py.File(path, "r")
        self.sur_time = jnp.array(data["domain"])

        self.modes = []
        for i in range(len(modelist)):
            self.modes.append(self.read_single_mode(data, modelist[i]))

    def read_function(self, node_data: h5py.Group) -> dict:
        try:
            result = {}
            if isinstance(node_data["n_nodes"], h5py.Dataset):
                n_nodes = int(node_data["n_nodes"][()])  # type: ignore
                result["n_nodes"] = n_nodes

                predictors = []
                for count in range(n_nodes):
                    try:
                        fit_data = node_data[
                            "node_functions/ITEM_%d/node_function/DICT_fit_data"
                            % (count)
                        ]
                    except ValueError:
                        raise ValueError("GPR Fit info doesn't exist")

                    assert isinstance(
                        fit_data, h5py.Group
                    ), "GPR Fit info is not a group"
                    res = h5Group_to_dict(fit_data)
                    node_predictor = EIMpredictor(res)
                    predictors.append(node_predictor)

                result["predictors"] = predictors
                result["eim_basis"] = jnp.array(node_data["ei_basis"])
                result["name"] = node_data["name"][()].decode("utf-8")  # type: ignore
                return result
            else:
                raise ValueError("n_nodes data doesn't exist")
        except ValueError:
            raise ValueError("n_nodes data doesn't exist")

    @staticmethod
    def make_empty_function(name: str, length: int) -> dict:
        return {
            "n_nodes": 1,
            "predictors": [lambda x: 1],
            "eim_basis": jnp.zeros((1, length)),
            "name": name,
        }

    def read_single_mode(self, file: h5py.File, mode: tuple[int, int]) -> dict:
        result = {}
        data = file["sur_subs/%s/func_subs" % (h5_mode_tuple[mode])]
        assert isinstance(data, h5py.Group), "Mode data is not a group"
        if mode == (2, 2):
            result["phase"] = self.read_function(data["ITEM_0"])  # type: ignore
            result["amp"] = self.read_function(data["ITEM_1"])  # type: ignore
        else:
            if mode[1] != 0:
                result["real"] = self.read_function(data["ITEM_0"])  # type: ignore
                result["imag"] = self.read_function(data["ITEM_1"])  # type: ignore
            else:
                local_function = self.read_function(data["ITEM_0"])  # type: ignore
                if local_function["name"] == "re":
                    result["real"] = local_function
                    result["imag"] = self.make_empty_function(
                        "im", local_function["eim_basis"].shape[1]
                    )
                else:
                    result["imag"] = local_function
                    result["real"] = self.make_empty_function(
                        "re", local_function["eim_basis"].shape[1]
                    )
        result["mode"] = mode
        return result


class NRSur7dq4DataLoader(eqx.Module):
    t_coorb: Float[Array, " n_sample"]
    t_ds: Float[Array, " n_dynam"]
    diff_t_ds: Float[Array, " n_dynam-1"]

    modes: list[dict]
    coorb: PolyPredictor

    @property
    def coorb_nmax(self) -> int:
        return self.coorb.n_max

    def __init__(
        self,
        path: str,
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
        data = h5Group_to_dict(h5py.File(path, "r"))
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

        self.coorb = self.read_coorb(data, coorb_nmax)

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
        return result

    def read_single_mode(self, file: dict, mode: tuple[int, int], n_max: int) -> dict:
        result = {}
        if mode[1] != 0:
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
            result["real"] = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_real"], n_max
            )
            result["imag"] = self.read_mode_function(
                file[f"hCoorb_{mode[0]}_{mode[1]}_imag"], n_max
            )

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
            coefs: jnp.ndarray, bfOrders: jnp.ndarray
        ) -> PolyPredictor:
            return make_polypredictor_ensemble(coefs, bfOrders, n_max)

        coefs = []
        bfOrders = []

        for i in range(len(self.t_ds) - 1):
            # The last time step is not included in the data
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
