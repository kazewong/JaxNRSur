---
title: 'JaxNRSur'
tags:
  - Python
  - Gravitational Waves
  - Numerical Relativity
  - Machine Learning
  - Jax
authors:
  - name: Kaze W. K. Wong
    orcid: 0000-0001-8432-7788
    affiliation: 1 
  - name: Maximiliano Isi
    orcid: 0000-0001-8830-8672
    affiliation: 2
  - name: Ethan Payne
    orcid: 0000-0003-4507-8373
    affiliation: 3
  - name: Vijay Varma
    orcid: 0000-0002-9994-1761
    affiliation: 4
affiliations:
  - name: Data Science and AI institute, Johns Hopkins University, Baltimore, MD 21218, US
    index: 1  
  - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY 10010, US
    index: 2
  - name: Department of Physics, California Institute of Technology, Pasadena, California 91125, USA
    index: 3
  - name: Department of Mathematics, Center for Scientific Computing and Data Science Research, University of Massachusetts, Dartmouth, MA 02747, USA
    index: 4
date: 23 June 2025
bibliography: paper.bib
---

# Summary

The detection and analysis of gravitational waves rely on accurate and efficient modeling of waveforms from compact binary coalescences. Numerical relativity surrogate models provide a way to interpolate between expensive numerical relativity simulations, enabling rapid and accurate waveform generation for parameter estimation and data analysis.

`JaxNRSur` is a Python library that implements numerical relativity surrogate models for gravitational waveforms using the JAX machine learning framework. By leveraging JAX's automatic differentiation, just-in-time (JIT) compilation, and accelerator support (GPU/TPU), `JaxNRSur` enables fast, differentiable, and scalable waveform generation. The package currently supports models such as NRHybSur3dq8 and NRSur7dq4, with a simple and extensible interface for users and developers.

Key features of `JaxNRSur` include:

## Key features

- **JAX-based implementation:** All waveform models are implemented using JAX, enabling automatic differentiation and seamless support for hardware accelerators.
- **JIT compilation:** Users can leverage JAX's JIT compilation (via Equinox's `filter_jit`) for significant speedups in waveform evaluation.
- **Vectorization:** Batch evaluation of waveforms is supported via `vmap`, allowing efficient computation over many parameter sets.
- **Extensible API:** The codebase is designed for extensibility, allowing new surrogate models to be added easily.
- **Automatic data management:** Required surrogate data is automatically downloaded and cached locally.
- **Simple interface:** Users can generate waveforms with minimal code, specifying only the time array and source parameters.

# Statement of need

Gravitational wave astronomy requires the evaluation of millions of waveform templates for parameter estimation and model selection. Numerical relativity simulations are the gold standard for waveform accuracy but are computationally expensive and impractical for direct use in data analysis. Surrogate models trained on these simulations provide a fast and accurate alternative, but existing implementations are often not differentiable, not optimized for accelerators, or lack a user-friendly interface.

`JaxNRSur` addresses these needs by providing:

- **Differentiable waveforms:** Thanks to JAX, gradients of waveforms with respect to source parameters or time can be computed automatically, enabling advanced inference and optimization techniques.
- **Accelerator support:** The same code runs efficiently on CPUs, GPUs, and TPUs, with no user intervention required.
- **Modern Python ecosystem:** Integration with JAX and Equinox allows users to leverage the latest advances in scientific computing and machine learning.
- **Ease of use:** A black-box interface is provided for typical users, while advanced users can extend or customize models as needed.

# Technical details

Waveform models in `JaxNRSur` are implemented as Python classes, parameterized by time and source parameters. The package uses Equinox to manage model parameters and ensure compatibility with JAX transformations. Users can JIT-compile or vectorize waveform evaluations for large-scale analyses. Automatic differentiation enables gradient-based sampling or optimization for parameter estimation.

Data required for the surrogate models is automatically downloaded from Zenodo and cached locally, ensuring reproducibility and ease of setup.

# References
