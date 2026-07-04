```@meta
CurrentModule = cDFT
```

# External Fields

External fields represent anything outside the fluid itself that the density profile
responds to: a solid wall, an atomistic surface, or (for electrolytes) a mean-field
electrostatic potential. They're passed to [`DFTSystem`](@ref cDFT.DFTSystem) alongside the
`model` and `structure`.

## Contents

```@contents
Pages = ["external_fields.md"]
Depth = 1
```

## Types and Constructors

```@docs
cDFT.Steele
cDFT.LJField
cDFT.ElectrostaticPotential
```
