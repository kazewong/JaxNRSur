import gwsurrogate
import jax.numpy as jnp

q = 3
incl = 0.14535
phiref = 2.625

chi1 = jnp.array([0.1, 0.5, 0.1])
chi2 = jnp.array([0.5, 0.1, 0.3])

params = jnp.concatenate([jnp.array([q]), chi1, chi2])

gwsurrogate.catalog.pull('NRSur7dq4')
sur = gwsurrogate.LoadSurrogate('NRSur7dq4')

# Note that I evaluate the waveform with gwsurrogate
t, h, dyn = sur(q, chi1, chi2, dt=1, f_low=0, inclination=incl, phi_ref=phiref, units='dimensionless', precessing_opts={'return_dynamics': True})

# Save the waveform and parameters to a file
jnp.savez('test_data.npz', t=t, h=h, dyn=dyn, params=params, incl=incl, phiref=phiref)
