from jaxNRSur.SurrogateModel import NRSur7dq4Model
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

params = jnp.array([1, 0, 0, 0, 0, 0, 0])

model = NRSur7dq4Model()
waveform_jax, dyn_jax = model.get_waveform(jnp.linspace(0, 1, 10), params, theta=0, phi=0) # note the time array here doesn't do anything 

import gwsurrogate

gwsurrogate.catalog.pull('NRSur7dq4')
sur = gwsurrogate.LoadSurrogate('NRSur7dq4')

t, h, dyn = sur(1, np.array([0,0,0]), np.array([0,0,0]), dt=1, f_low=0, inclination=0, phi_ref=np.pi/2, units='dimensionless', precessing_opts={'return_dynamics': True})

orbphase_gwsur = interp1d(t, dyn['orbphase'][:,], bounds_error=False, fill_value=0)(model.data.t_coorb)

plt.plot(model.data.t_coorb, dyn_jax[:,4]-orbphase_gwsur, label='Jax - gwsur')

plt.xlabel('time [M]')
plt.legend(loc='upper left')

plt.savefig(f"test_phase_comparison.pdf"); plt.clf()



plt.plot(model.data.t_coorb, 2*jnp.real(waveform_jax), label='Jax waveform')
plt.plot(t, jnp.real(h), label='GWSurrogate waveform')

plt.xlabel('time [M]')
plt.ylabel('strain')
plt.legend(loc='upper left')

plt.savefig(f"test_wf_comparison.pdf")