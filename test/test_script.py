from jaxNRSur.SurrogateModel import NRSur7dq4Model
import jax.numpy as jnp

# import matplotlib.pyplot as plt

params = jnp.array([1, 0, 0, 0, 0, 0, 0])

model = NRSur7dq4Model("/mnt/home/epayne/NRSur7dq4.h5")
data = model.get_waveform(jnp.linspace(0, 1, 10), params)
# plt.plot(model.data.t_coorb, jnp.real(data[0]))
# plt.savefig("test3.pdf")
