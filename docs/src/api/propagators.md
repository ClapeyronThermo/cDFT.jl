```@meta
CurrentModule = cDFT
```

## Contents

```@contents
Pages = ["propagators.md"]
Depth = 1
```

## Propagators

A propagator carries the connectivity between bonded beads of a chain molecule (or, for
[`IdealPropagator`](@ref cDFT.IdealPropagator), signals that a model has no chains at
all). It's how [`converge!`](@ref cDFT.converge!)'s fixed-point map turns each species'
own field into a chain-connectivity contribution to the functional derivative — for
SCFT (see [SCFT](../models/scft.md)), the same [`DiscreteGaussianChainPropagator`](@ref
cDFT.DiscreteGaussianChainPropagator) instead builds the forward/backward propagators used
to assemble density profiles directly.

```@docs
cDFT.IdealPropagator
cDFT.TangentHSPropagator
cDFT.DiscreteGaussianChainPropagator
```

## Functions

```@docs
cDFT.propagate!
cDFT.preallocate_propagator
```
