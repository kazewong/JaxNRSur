import gwsurrogate  # type: ignore
import numpy as np

seed = 124091248
size = 10

q = np.random.uniform(low=1, high=4, size=size)
incl = np.random.uniform(low=0, high=np.pi, size=size)
phiref = np.random.uniform(low=0, high=2 * np.pi, size=size)

chi1 = np.random.uniform(low=0.0, high=0.8, size=(size, 3))
# resample the chi1 which has magnitude > 0.8
mag = np.linalg.norm(chi1, axis=1)
while np.any(mag > 0.8):
    chi1[mag > 0.8] = np.random.uniform(low=0.0, high=0.8, size=(np.sum(mag > 0.8), 3))
    mag = np.linalg.norm(chi1, axis=1)

chi2 = np.random.uniform(low=0.0, high=0.8, size=(size, 3))
mag = np.linalg.norm(chi2, axis=1)
while np.any(mag > 0.8):
    chi2[mag > 0.8] = np.random.uniform(low=0.0, high=0.8, size=(np.sum(mag > 0.8), 3))
    mag = np.linalg.norm(chi2, axis=1)


gwsurrogate.catalog.pull("NRSur7dq4")
sur = gwsurrogate.LoadSurrogate("NRSur7dq4")


t = []
h = []
dyn = []
params = []
# Iterate over the waveforms
for i in range(size):
    print(i)
    params_i = np.array(
        [q[i], chi1[i][0], chi1[i][1], chi1[i][2], chi2[i][0], chi2[i][1], chi2[i][2]]
    )
    t_i, h_i, dyn_i = sur(
        q[i],
        chi1[i],
        chi2[i],
        f_low=0,
        inclination=incl[i],
        phi_ref=phiref[i],
        units="dimensionless",
        precessing_opts={"return_dynamics": True},
    )
    params.append(params_i)
    t.append(t_i)
    h.append(h_i)
    dyn.append(dyn_i)


# Save the waveform and parameters to a file
np.savez("test_data.npz", t=t, h=h, dyn=dyn, params=params, incl=incl, phiref=phiref)
