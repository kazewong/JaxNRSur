---
title: 'JaxNRSur - Numerical Relativity Surrogates in JAX'
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

The detection and analysis of gravitational waves relies on accurate and efficient modeling of waveforms from compact binary coalescences. Numerical relativity (NR) surrogate models are a family of data-driven models that are directly trained on NR simulations to provide fast and accurate approximations of these waveforms without many assumptions about the form of the model.

`JaxNRSur` is a Python library that implements NR waveform models in `JAX` [@jax2018github], a high-performance numerical computing library that supports automatic differentiation, just-in-time (JIT) compilation, and hardware acceleration (GPU/TPU). By using JAX, `JaxNRSur` provides the following key features:

## Key features

- **Automatic differentiation:** Compute gradients of waveforms with respect to time, source parameters, or model parameters (i.e., the data in the surrogate model) automatically, enabling advanced inference techniques such as gradient-based sampling or optimization.
- **Accelerator support:** The same code runs efficiently on CPUs, GPUs, and TPUs, with no user intervention required. This allows for fast waveform evaluations in large-scale analyses.
- **Simple vectorization:** JAX's vectorization capabilities allow users to efficiently evaluate waveforms for multiple source parameters or times in parallel.
- **Unified interface:** A consistent and user-friendly interface for accessing different NR surrogate models, making it easy to switch between models.

`JaxNRSur` currently includes implementations of several NR surrogate models, including:

- `NRHybSur3dq8`: A hybridized surrogate model for non-precessing binary black hole mergers [@Varma:2018mmi], valid for mass ratios up to 8.
- `NRSur7dq4`: A surrogate model for precessing binary black hole mergers [@Varma:2019csw], valid for mass ratios up to 4.

We have validated our implementation against the original implementations of these models in `gwsurrogate` [@Field:2025isp] to ensure correctness and consistency. The package also includes utilities for downloading and caching the required data files from Zenodo [@https://doi.org/10.25495/7gxk-rd71], making it easy to set up and use.

# Statement of need

Gravitational wave astronomy requires the evaluation of millions of waveform templates for parameter estimation and model selection. Numerical relativity simulations are the gold standard for waveform accuracy but are computationally expensive and impractical for direct use in data analysis. Surrogate models trained on these simulations provide a fast and accurate alternative, but existing implementations are often not differentiable, not optimized for accelerators, or lack a user-friendly interface.

`JaxNRSur` addresses these needs by providing:

- **Differentiable waveforms:** Thanks to JAX, gradients of waveforms with respect to source parameters or time can be computed automatically, enabling advanced inference and optimization techniques.
- **Accelerator support:** The same code runs efficiently on CPUs, GPUs, and TPUs, with no user intervention required.
- **Modern Python ecosystem:** Integration with JAX and Equinox allows users to leverage the latest advances in scientific computing and machine learning.
- **Ease of use:** A black-box interface is provided for typical users, while advanced users can extend or customize models as needed.

# Acknowledgements

We thank Tousif Islam, Will Farr, Keefe Mitman, and Colm Talbot for their contributions to the JaxNRSur codebase and discussions that improved the package.
V.V.~acknowledges support from NSF Grant No. PHY-2309301 and UMass Dartmouth's
Marine and Undersea Technology (MUST) Research Program funded by the Office of
Naval Research (ONR) under Grant No. N00014-23-1–2141.

# References
