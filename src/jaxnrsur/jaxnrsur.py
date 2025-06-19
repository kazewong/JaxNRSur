    def get_waveform_td(
        self,
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
        GMSUN_SI = 1.32712442099000e20
        C_SI = 2.99792458000000e08
        RSUN_SI = GMSUN_SI / C_SI**2

        # parsecs to SI
        PC_SI = 3.08567758149136720000e16
        MPC_SI = 1e6 * PC_SI

        # form time array with desired sampling rate and duration
        # N = int(seglen*srate)
        # time = jnp.arange(N)/srate - seglen + 2

        # evaluate the surrogate over the equivalent geometric time
        time_m = time * C_SI / RSUN_SI / mtot
        hrM_p, hrM_c = self.get_waveform_geometric(time_m, jnp.array(params[2:]))

        if alpha_window > 0:
            # create a window for the waveform: the form of the window
            # is chosen such that it is 0 at the start, as well as zero
            # first and second derivative at the start, and is 1 and zero
            # derivatives at the end.
            Tcoorb = self.data.t_coorb[-1] - self.data.t_coorb[0]

            window_start = jnp.max(jnp.array([time_m[0], self.data.t_coorb[0]]))
            window_end = window_start + alpha_window * Tcoorb

            x = (time_m - window_start) / (window_end - window_start)

            window = jnp.select(
                [time_m < window_start, time_m > window_end],
                [0.0, 1.0],
                default=x * x * x * (10 + x * (6 * x - 15)),
            )
            hrM_p *= window
            hrM_c *= window

        # this is h * r / M, so scale by the mass and distance
        const = mtot * RSUN_SI / dist_mpc / MPC_SI
        return hrM_p * const, hrM_c * const

    def get_waveform_fd(
        self,
        time: Float[Array, " n_sample"],
        params: Float[Array, " n_param"],
        alpha_window: float = 0.1,
    ) -> tuple[Float[Array, " n_freq"], Float[Array, " n_freq"]]:
        """
        Get the waveform in the frequency domain.
        """
        # this could be moved inside get_waveform_td if
        # we change the API to have seglen and delta_t as
        # input, rather than time
        # N = int(seglen/delta_t)
        # time = jnp.arange(N)*delta_t - seglen + 2

        hp_td, hc_td = self.get_waveform_td(time, params, alpha_window=alpha_window)

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
