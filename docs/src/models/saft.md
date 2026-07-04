```@meta
CurrentModule = cDFT
```

# SAFT-based Models

cDFT's most complete family of free-energy functionals, built as a Rosenfeld FMT
hard-sphere reference plus model-specific dispersion, association, chain and polar terms.
The bulk equation of state for each of these is obtained directly from
[Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) — cDFT only supplies the
inhomogeneous (weighted-density) functional built on top of it.

## Contents

```@contents
Pages = ["saft.md"]
Depth = 1
```

## PC-SAFT Family

`PCSAFT` and `PCPSAFT` use a Weighted Density Functional approach (Sauer & Gross, 2017) and
do not require a chain propagator. `HomogcPCPSAFT`/`HeterogcPCPSAFT` extend this to
group-contribution chains (see [Group-Contribution & Heterosegmented Chains](@ref)),
`QPCPSAFT` adds quadrupolar interactions, and `pharmaPCSAFT` targets pharmaceutical
solutes.

```@docs
cDFT.PCSAFT
cDFT.PCPSAFT
cDFT.HomogcPCPSAFT
cDFT.HeterogcPCPSAFT
cDFT.QPCPSAFT
cDFT.pharmaPCSAFT
```

## Mie-Potential SAFT Models

```@docs
cDFT.SAFTVRMie
cDFT.SAFTgammaMie
```
