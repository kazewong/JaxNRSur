# JaxNRSur
Numerical relativity surrogate waveform in Jax

<!-- <a href="https://jaxnrsur.readthedocs.io/en/latest/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a> -->
<a href="https://github.com/kazewong/JaxNRSur/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="doc"/>
</a>

# Quickstart

## Installation

The recommended way to install `jaxNRSur` is via `uv`. `uv` is a python package and project manager that takes inspiration and is written in `rust`. You can find the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
Once you have `uv` installed, you can install `jaxNRSur` with`uv add JaxNRSur` in the project you are developing.
If you want to try this package out, clone this repository and cd into the directory, then run `uv sync --dev` should produce an environment in the directory. The environment should have `.venv/bin/activate` which you can run `source .venv/bin/activate` to activate the environment.

Alternatively, you can install `jaxNRSur` with `pip`: `pip install JaxNRSur`

If you want to use GPU, you will need to run `uv sync --all-extras` or `pip install -U "jax[cuda12]"` to install the version of `jax` which is compatible with an Nvidia GPU.

## Basic Usage
`jaxNRSur` has a pretty simple interface. At its core, a surrogate waveform is parameterized as `h(t, theta)`, where `h` is the strain, `t` is the time sample array, `theta` is the gravitational wave source parameters such as the masses and spins.

Right now, `jaxNRSur` supports the following models:
- NRHybSur3dq8Model
- NRSur7dq4Model

```python
import jax.numpy as jnp
from jaxnrsur.NRHybSur3dq8 import NRHybSur3dq8Model
from jaxnrsur.NRSur7dq4 import NRSur7dq4Model

time = jnp.linspace(-1000, 100, 100000)
params = jnp.array([0.9, 0.1, 0.1])
model = NRHybSur3dq8Model()
h = model(time, params)
params = jnp.array([0.9, 0.0, 0.5, 0.0, 0.5, 0.0, 0.3])
model = NRSur7dq4Model()
h = model(time, params)
```

## Jax features

### JIT Compilation

`jax` not only support JIT compilation like `numba`, but it also do so in an accelerator-aware manner, meaning once we have developed the source code in `jax`, it is immediately compatible with accelerators such as GPUs.
To use JIT to speed up the code, all you have to do is the following:

```python
#Let's use NRSur7dq4 as an example
model = NRHybSur3dq8Model()
jitted_model = eqx.filter_jit(model)
h = jitted_model(time, params)
```

Note that since our `NRHybSur3dq8` contains some parameters that are not compatible with `jax` JIT tranformation, we built the package on top of `equinox`, which is a JAX-compatible library for building neural networks. It allows us to write object oriented code that knows how to handle the parameters associated with each model without the need of writing purely functional code. Instead of using the default `jax` JIT transformation `jax.jit`, we use `equinox.filter_jit`, which seperate out parameters that are not compatible with `jax.jit` before it runs the transformation under the hood.

### Automatic Differentiation

The next feature offered by `jax` is automatic differentiation. This allows us to compute the gradient of a function with respect to its input parameters. This is the corner stone of deep learning, as it allows us to use gradient descent to optimize the parameters of a model. In our case, for people who are interested in parameter estimation, perhaps one may want to use the gradient of the posterior function with respect to the parameters. For people who are interested in optimizing the waveform parameters, one can compute the gradient of the waveform with respect to the parameters. Here we give two examples of building the gradient functions for `NRHybSur3dq8`, one with respect to the time grid, and one with respect to the parameters.

```python
def target(time, params):
  return jnp.sum(model(time, params))

grad_target_time = jax.grad(target, argnums=0)
grad_target_params = jax.grad(target, argnums=1)
```

### Vectorization

`jax` offers the use of `vmap` to vectorize functions. This is different from a for loop considering `jax` will fuse some of the operations under the hood to achieve better performance. On a CPU this makes a small but noticible differences since modern day CPUs offers some vectorization capabilities, but where this really shines is on accelerators. From some initial microbenchmarks on a 5070Ti (16GB VRAM), the `NRSur7dq4` waveform can be evaluated for 500 parameters in ~300 ms.

Similar to JIT, we need to use `eqx.filter_vmap` instead of `jax.jit` in our case. Another note is that `vmap` does not compile the code, so one needs to use `eqx.filter_jit` on top of `eqx.filter_vmap` to compile the code.

```python
params = jnp.array([[0.9, 0.1, 0.1]])
params_multi = jnp.repeat(params, 10, axis=0)
h_multi = eqx.filter_jit(eqx.filter_vmap(model.get_waveform, in_axes=(None, 0)))(
    time, params_multi
)
```

## Local data cache

If the data is not already downloaded, then this package will look for the data on `Zenodo` and download it into `$HOME/.jaxNRSur`.
If the data is already downloaded, then the package will reuse the cached data

<!-- # Benchmark

Add benchmarking results here. -->

# Attribution

Coming soon. For now, give us a star and keep an eye out!
