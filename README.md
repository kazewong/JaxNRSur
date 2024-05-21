# JaxNRSur
Numerical relativity surrogate waveform in Jax

# Quickstart

## Installation
`pip install JaxNRSur`

If you want to use GPU, you will need to install the CUDA version of jax as well with
`pip install -U "jax[cuda12]"`

## Basic Usage
`jaxNRSur` has a pretty simple interface. At its core, a surrogate waveform is parameterized as `h(t, theta)`, where `h` is the strain, `t` is the time sample array, `theta` is the gravitational wave source parameters such as the masses and spins.


```python
import jax.numpy as jnp
from jaxNRSur.SurrogateModel import NRHybSur3dq8Model
time = jnp.linspace(-1000, 100, 100000)
params = jnp.array([0.9, 0.1, 0.1])
model = NRHybSur3dq8Model("./NRHybSur3dq8.h5")
h = model(time, params)
```

Soon, we will provide an interface for the user to interact with the surrogate data as well.

## Surrogate data
At some point, we will add the capability to download the data from Zenodo to the data loader, but for the time being, download the relevant data from here:

NRHybSur3dq8.h5:
https://zenodo.org/records/3348115/files/NRHybSur3dq8.h5?download=1

NRSur7dq4.h5:
https://zenodo.org/records/3348115/files/NRSur7dq4.h5?download=1

# Benchmark

Add benchmarking results here.

# Attribution

