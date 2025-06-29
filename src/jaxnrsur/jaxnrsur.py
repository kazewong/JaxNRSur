import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxnrsur.DataLoader import DataLoader
from abc import abstractmethod


# geometric units to SI
GMSUN_SI = 1.32712442099000e20
C_SI = 2.99792458000000e08
RSUN_SI = GMSUN_SI / C_SI**2

# parsecs to SI
PC_SI = 3.08567758149136720000e16
MPC_SI = 1e6 * PC_SI


class WaveformModel:
    data: DataLoader

    @abstractmethod
    def get_waveform_geometric(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_param"],
        theta: float,
        phi: float,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        raise NotImplementedError


class JaxNRSur:
    model: WaveformModel
    alpha_window: float = 0.1

    def __init__(self, model: WaveformModel, alpha_window: float = 0.1):
        self.model = model
        self.alpha_window = alpha_window

    def window_function(
        self,
        t: Float[Array, " n_sample"],
        hp: Float[Array, " n_sample"],
        hc: Float[Array, " n_sample"],
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        # create a window for the waveform: the form of the window
        # is chosen such that it is 0 at the start, as well as zero
        # first and second derivative at the start, and is 1 and zero
        # derivatives at the end.
        Tcoorb = self.model.data.sur_time[-1] - self.model.data.sur_time[0]

        window_start = jnp.max(jnp.array([t[0], self.model.data.sur_time[0]]))
        window_end = window_start + self.alpha_window * Tcoorb

        x = (t - window_start) / (window_end - window_start)

        window = jnp.select(
            [t < window_start, t > window_end],
            [0.0, 1.0],
            default=x * x * x * (10 + x * (6 * x - 15)),
        )
        hp *= window
        hc *= window
        return hp, hc

    # # window surrogate start with a window that is 0 at the start, as well as zero
    # # first and second derivative at the start, and is 1 and zero derivatives
    # # at the end, i.e., x^3(10 + x(6x - 15))
    # t = self.data.t_coorb - self.data.t_coorb[0]
    def get_waveform_td(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_param"],
        theta: float = 0.0,
        phi: float = 0.0,
    ) -> tuple[Float[Array, " n_sample"], Float[Array, " n_sample"]]:
        """
        Get the waveform in the time domain in SI units.
        """
        # get scaling parameters
        mtot = params[0]
        dist_mpc = params[1]

        # form time array with desired sampling rate and duration
        # N = int(seglen*srate)
        # time = jnp.arange(N)/srate - seglen + 2

        # evaluate the surrogate over the equivalent geometric time
        time_m = time * C_SI / RSUN_SI / mtot
        hrM_p, hrM_c = self.model.get_waveform_geometric(
            time_m, jnp.array(params[2:]), theta, phi
        )

        if self.alpha_window > 0:
            hrM_p, hrM_c = self.window_function(time_m, hrM_p, hrM_c)

        # this is h * r / M, so scale by the mass and distance
        const = mtot * RSUN_SI / dist_mpc / MPC_SI
        return hrM_p * const, hrM_c * const

    def get_waveform_fd(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_param"],
    ) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"]]:
        """
        Get the waveform in the frequency domain.
        """
        # this could be moved inside get_waveform_td if
        # we change the API to have seglen and delta_t as
        # input, rather than time
        # N = int(seglen/delta_t)
        # time = jnp.arange(N)*delta_t - seglen + 2

        hp_td, hc_td = self.get_waveform_td(time, params)

        h_fd = jnp.fft.fft(hp_td - 1j * hc_td)
        # f = jnp.fft.fftfreq(N, delta_t)

        # obtain hp_fd and hc_fd
        # rolling the arrays to get the positive and negative frequency components
        # aligned correctly, as in np.fft.rfft
        n = len(h_fd) // 2 + 1
        h_fd_positive = h_fd[:n]
        conj_h_fd_negative = jnp.conj(jnp.fft.ifftshift(h_fd))[:n][::-1]
        hp_fd = (h_fd_positive + conj_h_fd_negative) / 2
        hc_fd = 1j * (h_fd_positive - conj_h_fd_negative) / 2
        return hp_fd, hc_fd
