# JaxNRSur
Numerical relativity surrogate waveform in Jax

<!-- <a href="https://jaxnrsur.readthedocs.io/en/latest/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a> -->
<a href="https://github.com/kazewong/JaxNRSur/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="doc"/>
</a>

<a href='https://coveralls.io/github/kazewong/JaxNRSur?branch=main'>
  <img src='https://coveralls.io/repos/github/kazewong/JaxNRSur/badge.svg?branch=main' alt='Coverage Status' />
</a>

# Quickstart

## Installation

The recommended way to install `jaxNRSur` is via `uv`. [uv](https://docs.astral.sh/uv/) is a python package and project manager that takes inspiration and is written in `rust`. You can find the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
Once you have `uv` installed, you can install `JaxNRSur` with`uv add JaxNRSur` in the project you are developing.


If you want to try this package out, run the follow commands to create and activate a development environment
```
git clone https://github.com/kazewong/JaxNRSur
cd JaxNRSur
uv sync --dev
source .venv/bin/activate
```

Alternatively, you can install `jaxNRSur` with `pip install JaxNRSur`

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

This interface generate the waveform that does not contains extransic scaling with the total mass and distance. To obtain a dimensionful waveform, we provide a thin wrapper on top of the core API. Say you have constructed a model, you can then get dimensionful waveform with 

```python
from jaxnrsur import JaxNRSur

model = NRHybSur3dq8Model()
jaxnrsur = JaxNRSur(model, alpha_window=0.1, segment_length=4.0, sampling_rate=4096)
params_dimensionful = jnp.array([60.0, 400.0, 0.1, 0.2, 0.9, 0.1, 0.1])
h = jaxnrsur.get_waveform_td(time, params_dimensionful)
```
The dimensionful parameters assume the first four parameters to be `[M_tot, distance, inclination, phase_c]`, followed by the dimensionless parameters.

This higher level API also provides functionality to generate frequency domain waveforms by transforming the time domain waveform into frequency domain. You can find more detail [here](https://github.com/kazewong/JaxNRSur/blob/4d11d10df27fe242ef0859335bad7b854387f502/src/jaxnrsur/__init__.py#L130).

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
  return jnp.sum(model(time, params)).real

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

## Tutorial notebooks

Considering this package to be fairly lightweight, we think the combination of docstrings and notebook should suffice as documentations. You can find some examples for using the package in the `example` directory. Please open an issue (or better, a PR!) if you think there are extra use cases that could benefit from more documentation.

## Benchmark

<!-- Add a notebook on google colab to show the benchmark. -->

This is a benchmarking notebook hosted on Google Colab for people who want to try using `JaxNRSur` with a GPU. Note that the performance of the waveform on Colab depends on the provisioning of the machine, which is out of our control. In general, if you get a T4 GPU on colab, the run time should be somewhere around 60ms for 100 evaluation as parameterized and shown on the notebook.

<a href="https://colab.research.google.com/drive/1A12tzSPdFBL_jzWYLfll4yB1H2iWRtoi?usp=sharing">
<img alt="Static Badge" src="https://img.shields.io/badge/Colab-benchmark-orange?style=for-the-badge&logo=googlecolab">
</a>

## Local data cache

If the data is not already downloaded, then this package will look for the data on `Zenodo` and download it into `$HOME/.jaxNRSur`.
If the data is already downloaded, then the package will reuse the cached data

<!-- # Benchmark

Add benchmarking results here. -->

# Attribution

Coming soon. For now, give us a star and keep an eye out!
