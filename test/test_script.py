from jaxNRSur.SurrogateModel import NRSur7dq4Model
import jax.numpy as jnp

import matplotlib.pyplot as plt

params = jnp.array([1, 0, 0, 0, 0, 0, 0])

model = NRSur7dq4Model()
data = model.get_waveform(jnp.linspace(0, 1, 10), params)

for i in range(21):
    plt.plot(model.data.t_coorb, jnp.real(data[:,i]))

    plt.savefig(f"test3_{i}.pdf")

print(data[:,0])
print(data.shape)
