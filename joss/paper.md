---
title: "JaxNRSur - Numerical Relativity Surrogates in JAX"
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

The detection and analysis of gravitational waves rely on accurate and
efficient modeling of waveforms from compact binary coalescences. Numerical
relativity (NR) surrogate models are a family of data-driven models that are
directly trained on NR simulations to provide fast and accurate approximations
of these waveforms without many assumptions about the form of the model.

`JaxNRSur` is a Python library that implements NR waveform models in `JAX`
[@jax2018github], a high-performance numerical computing library that supports
automatic differentiation, just-in-time (JIT) compilation, and hardware
acceleration (GPU/TPU). By using JAX, `JaxNRSur` provides the following key
features:

## Key features

- **Accelerator support:** The same code runs efficiently on CPUs, GPUs, and TPUs, with no user intervention required. This allows for fast waveform evaluations in large-scale analyses.
- **Simple vectorization:** JAX's vectorization capabilities allow users to efficiently evaluate waveforms for multiple source parameters or times in parallel.
- **Automatic differentiation:** Compute gradients of waveforms with respect to time, source parameters, or model parameters (i.e., the data in the surrogate model) automatically, enabling advanced inference techniques such as gradient-based sampling or optimization.
- **Unified interface:** A consistent and user-friendly interface for accessing different NR surrogate models, making it easy to switch between models.

`JaxNRSur` currently includes implementations of several NR surrogate models,
including:

- `NRHybSur3dq8`: A hybridized surrogate model for non-precessing binary black
  hole mergers [@Varma:2018mmi], trained on mass ratios up to 8.
- `NRSur7dq4`: A surrogate model for precessing binary black hole mergers [@Varma:2019csw], trained on mass ratios up to 4.

We have validated our implementation against the original implementations of
these models in `gwsurrogate` [@Field:2025isp] to ensure correctness and
consistency. The package also includes utilities for downloading and caching the
required data files from Zenodo [@https://doi.org/10.25495/7gxk-rd71], making it
easy to set up and use.

# Statement of need

The original implementations of NR surrogate models in `gwsurrogate`
[@Field:2025isp] are mainly implemented in `numpy` and `scipy` with additional
bindings from `c`, which gives the package reasonable performance. However,
this implementation does not take advantage of more modern computing paradigms
such as the use of accelerators (GPUs/TPUs) and automatic differentiation, which
have been widely adopted in the machine learning and high-performance computing
communities. This presents a few challenges when using and developing NR
surrogate models, and this package aims to address these challenges:

<!-- How accelerator comes into play -->

A main challenge in using NR surrogate models for downstream tasks such as
parameter estimation is the relatively high computational cost compared to other
waveform approximants such as IMRPhenomXPHM [@Pratten:2020ceb]. NR surrogate
models involve a lot of dense matrix multiplications and other linear algebra
operations, which can be efficiently parallelized on accelerators such as GPUs
and TPUs. This `JAX` implementation allows users to leverage accelerators to
speed up waveform evaluations. For example, in a publicly available benchmark on
an NVIDIA T4 GPU in a Google Colab environment, the `NRSur7dq4` waveform can be
evaluated for 100 parameters in approximately 65 ms, which is significantly
faster than the original implementation in `gwsurrogate`[^1]. This speedup is
further compounded by the fact that `JAX` supports vectorization of functions
that is accelerator-aware, providing the performance the NR surrogate model
family needs for downstream tasks.

[^1]: Benchmarking results on Google Colab may vary depending on the specific hardware and software environment that is provisioned.

<!-- How differentiability can be used -->

<!-- Gradient on source parameters -->

Another feature this package provides is the ability to compute gradients of the
waveform with respect to the source parameters through automatic
differentiation. This is useful for tasks such as template bank generation
[@Coogan:2022qxs] or gradient-based Markov chain Monte Carlo (MCMC) sampling
[@Betancourt2017ACI] [@Wong:2022xvh] [@cabezas2024blackjax]. Parallel to the
improvement in waveform evaluation throughput, being able to leverage gradient
information often speeds up the convergence of these downstream tasks, which
further improves the performance of a complete pipeline when compared to
non-gradient-based counterparts. On top of performance gains, higher-order
derivatives such as Hessians can be used as a natural metric for understanding
uncertainties in the pipeline and help in sensitivity analysis.

<!-- Gradient on model parameters -->

Finally, this package also supports differentiating with respect to the model parameters
in a fashion similar to how neural networks are trained. This allows users to
fine-tune the model parameters to better fit their data [@Lam:2023oga]. Another
research avenue our package opens is more flexible ways to incorporate
uncertainties into the models. Similar to how neural networks can be promoted to
probabilistic models through techniques such as Bayesian neural networks
[@Jospin2020HandsOnBN], the gradient information allows users to more
efficiently fit a large number of model parameters, enabling the possibility of
promoting each model parameter to a distribution rather than a point estimate.
This can be useful for understanding systematics and uncertainties in the
models, which is crucial for accurate gravitational wave analyses.

# Acknowledgements

We thank Tousif Islam, Will Farr, Keefe Mitman, and Colm Talbot for their
contributions to the JaxNRSur codebase and discussions that improved the
package. V.V.~acknowledges support from NSF Grant No. PHY-2309301 and UMass
Dartmouth's Marine and Undersea Technology (MUST) Research Program funded by the
Office of Naval Research (ONR) under Grant No. N00014-23-1–2141.

# References
