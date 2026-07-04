```@meta
CurrentModule = cDFT
```
# cDFT.jl

cDFT intends to provide a comprehensive library of classical Density Functional Theory
(cDFT) and Self-Consistent Field Theory models, as well as a simple framework to develop
your own! cDFT is built directly on top of [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl)
and reuses its equations of state as the bulk free-energy model underlying every
inhomogeneous calculation — the two packages are meant to be used together.

With cDFT you can compute density profiles, surface/interfacial tensions and adsorption
isotherms for fluids next to walls, in pores, around solutes, at vapour-liquid and
liquid-liquid interfaces, in microphase-separated copolymer melts, and for electrolytes
near charged surfaces — in 1D, 2D or 3D, on the CPU or GPU, and (via Dynamic DFT) as a
function of time as well as space.

The documentation is laid out as follows:

- **[Installation](@ref)**: how to install cDFT.jl and Clapeyron.jl together.
- **Tutorials**: worked examples of increasing complexity, starting from a single fluid
  next to a wall and building up to electrolytes, dynamic DFT and GPU acceleration.
- **Available Models**: reference for every free-energy functional cDFT provides, grouped
  by family.
- **Structures & External Fields**: reference for every geometry and external field type.
- **API**: lower-level reference for how a [`DFTSystem`](@ref cDFT.DFTSystem) is composed
  and solved.
- **[FAQ](@ref)**: common gotchas.

If you're new to cDFT, start with the [Getting Started](@ref) tutorial.

### Authors

- [Pierre J. Walker](mailto:pjwalker@caltech.edu), California Institute of Technology
- [Andrés Riedemann](mailto:andres.riedemann@gmail.com), University of Concepción

### License

cDFT.jl is licensed under the [MIT license](https://github.com/ClapeyronThermo/cDFT.jl/blob/main/LICENSE.md).

### Citing cDFT.jl

cDFT.jl does not yet have a dedicated publication — for now, please cite the
[GitHub repository](https://github.com/ClapeyronThermo/cDFT.jl) directly. Since every cDFT
calculation is built on top of a Clapeyron.jl bulk equation of state, please also cite
[Clapeyron.jl](https://pubs.acs.org/doi/10.1021/acs.iecr.2c00326) itself, along with the
specific equation of state used (obtainable via `Clapeyron.cite(model)`) and, where
relevant, the original cDFT functional reference (e.g. Sauer & Gross, 2017 for the
weighted-density PC-SAFT functional; see each entry under **Available Models** for its
reference).
