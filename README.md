# JaxNRSur
Numerical relativity surrogate waveform in Jax

<a href="https://jaxnrsur.readthedocs.io/en/latest/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/kazewong/JaxNRSur/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="doc"/>
</a>

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
model = NRHybSur3dq8Model()
h = model(time, params)
```

Soon, we will provide an interface for the user to interact with the surrogate data as well.

## Surrogate data

The data needed by the surrogate model will be downloaded and stored at `$HOME/.jaxNRSur`, and if it is already downloaded, then it will reuse the cached data

# Benchmark

Add benchmarking results here.

# Attribution

