```@meta
CurrentModule = cDFT
```

# Structures

Every [`DFTSystem`](@ref cDFT.DFTSystem) needs a `structure`: the geometry, grid resolution,
conditions and bulk densities the density profile is computed on. This page lists every
concrete structure type. See [Choosing a Geometry & Adsorption](@ref) for guidance on which one to reach for.

## Contents

```@contents
Pages = ["structures.md"]
Depth = 1
```

## Single-Phase Structures

Used for a single bulk phase, typically next to an [external field](@ref "External Fields")
(a wall, a pore, a solute) or on its own as a bulk-consistency check.

```@docs
cDFT.Uniform1DCart
cDFT.Uniform2DCart
cDFT.Uniform3DCart
cDFT.Uniform1DSphr
cDFT.Uniform1DCyl
cDFT.ExternalField1DCart
```

## Two-Phase / Interface Structures

Used to resolve an interface between two bulk phases (vapour-liquid, liquid-liquid, or a
microphase-separated copolymer melt), initialised as a sigmoidal density profile between
the two supplied bulk densities.

```@docs
cDFT.TwoPhase1DCart
cDFT.TwoPhase2DLamCart
cDFT.TwoPhase3DLamCart
cDFT.TwoPhase2DHexCart
cDFT.TwoPhase3DHexCart
cDFT.TwoPhase3DSphrCart
```

## Functions

```@docs
cDFT.initialize_profiles
```
