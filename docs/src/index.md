---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: cDFT.jl
  text: Classical Density Functional Theory in Julia
  image:
    src: cDFT_logo.svg
    alt: cDFT.jl
  tagline: A comprehensive, extensible library of classical DFT and Self-Consistent Field Theory models, built directly on top of Clapeyron.jl
  actions:
    - theme: brand
      text: Getting started
      link: /tutorials/getting_started
    - theme: alt
      text: Installation
      link: /installation
    - theme: alt
      text: View on GitHub
      link: https://github.com/ClapeyronThermo/cDFT.jl

features:
  - icon: ⚛️
    title: Many free-energy functionals
    details: PC-SAFT, SAFT-VR Mie, SAFT-γ Mie, COFFEE, electrolytes and more, all sharing Clapeyron.jl's equations of state as the underlying bulk model
    link: /models/saft

  - icon: 🧱
    title: Flexible geometries
    details: Planar, cylindrical, spherical and fully 3D structures, plus block-copolymer microphase morphologies (BCC, gyroid, hex, lamellar)
    link: /structures

  - icon: ⚡
    title: GPU acceleration
    details: The same model code runs unchanged on CPU or GPU via KernelAbstractions.jl and Enzyme.jl
    link: /tutorials/gpu_acceleration

  - icon: ⏱️
    title: Dynamic DFT
    details: Evolve density profiles in time as well as space, to watch phase separation and microphase ordering actually happen
    link: /tutorials/dynamic_dft
---

## What is this?

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

## Quick start guide

### Installation

```julia
using Pkg
Pkg.add(["cDFT", "Clapeyron"])
```

See [Installation](@ref) for optional extensions (plotting, GPU, group-contribution
connectivity, Dynamic DFT).

### A first calculation

```julia
using Clapeyron, cDFT

model = PCSAFT(["methane"])
T, p = 150.0, 1e7
v = Clapeyron.volume(model, p, T, [1.0]; phase=:liquid)
ρbulk = [1/v]
L = cDFT.length_scale(model)

width = 5L
surface = Steele(["graphite"], width)
structure = Uniform1DCart((p, T), ρbulk, [0.5L, width-0.5L], 201)

system = DFTSystem(model, structure, surface)
ρ = initialize_profiles(system)
converge!(system, ρ)
```

`ρ` is now the converged density profile of liquid methane next to a graphite wall. See
[Getting Started](@ref) for the full walkthrough, including plotting the result.

## Citing cDFT.jl

cDFT.jl does not yet have a dedicated publication — for now, please cite the
[GitHub repository](https://github.com/ClapeyronThermo/cDFT.jl) directly. Since every cDFT
calculation is built on top of a Clapeyron.jl bulk equation of state, please also cite
[Clapeyron.jl](https://pubs.acs.org/doi/10.1021/acs.iecr.2c00326) itself, along with the
specific equation of state used (obtainable via `Clapeyron.cite(model)`) and, where
relevant, the original cDFT functional reference (e.g. Sauer & Gross, 2017 for the
weighted-density PC-SAFT functional; see each entry under [Available Models](@ref "SAFT-based Models")
for its reference).

## Related packages

- [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) provides every bulk
  equation of state cDFT builds its inhomogeneous functionals on top of, and is required
  alongside cDFT for essentially all use.

## Authors

- [Pierre J. Walker](mailto:pjwalker@caltech.edu), California Institute of Technology
- [Andrés Riedemann](mailto:andres.riedemann@gmail.com), University of Concepción

## License

cDFT.jl is licensed under the [MIT license](https://github.com/ClapeyronThermo/cDFT.jl/blob/main/LICENSE.md).
