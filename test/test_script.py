from jaxNRSur.SurrogateModel import NRSur7dq4Model
import jax.numpy as jnp

import matplotlib.pyplot as plt

params = jnp.array([0.3, 0.6, 0, 0, 0, -0.7, 0])

model = NRSur7dq4Model()
data = model.get_waveform(jnp.linspace(0, 1, 10), params, theta=jnp.pi/2, phi=jnp.pi/2)


plt.plot(model.data.t_coorb, jnp.real(data), label='q=0.3, S1x=0.6, S2y=-0.7, edge-on')

plt.xlabel('time [M]')
plt.ylabel('Strain [rh/M]')
plt.legend(loc='upper left')

plt.savefig(f"test_full2.pdf")

print(data)
print(data.shape)
