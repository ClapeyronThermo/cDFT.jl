```@meta
CurrentModule = cDFT
```

## Contents

```@contents
Pages = ["free_energy.md"]
Depth = 1
```

## Free Energy Evaluation

Low-level functions used internally by [`converge!`](@ref cDFT.converge!) and the
[Methods](methods.md) to evaluate the free-energy functional and its density derivative
for a given density profile. Not typically called directly by users.

```@docs
cDFT.free_energy
cDFT.δFδρ_res
cDFT.F_res
cDFT.F_ideal
```
