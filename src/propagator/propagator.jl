"""
    IdealPropagator <: DFTPropagator
    Type used to indicate that the propagator is ideal.

# Description
Ideal propagator for DFT calculations. Assumes all species are represented by a single bead.
"""
struct IdealPropagator <: DFTPropagator end
"""
    TangentHSPropagator <: DFTPropagator
    Type used to indicate that the propagator is a Tangent Hard-Sphere propagator.

# Description
Tangent Hard-Sphere propagator for DFT calculations. Assumes all species are made up of tangentially-bonded hard-sphere beads. Contains:
- `map`: The Fourier transform of the weights used in the propagator.

Uses the algorithm developed by Xu et al. (2009) to handle branching.

# References
1. Xu, X., Cao, D., Zhang, X. and Wang W. (2009). Universal version of density-functional theory for polymers with complex architecture. PHYSICAL REVIEW E, 79, 021805. [doi::10.1103/PhysRevE.79.021805](https://doi.org/10.1103/PhysRevE.79.021805)
"""
struct TangentHSPropagator{M} <: DFTPropagator 
    map::M
end

function propagate!(system::AbstractcDFTSystem, δf_res, ρ, cache_propagator)
    if !(system.propagator isa IdealPropagator)
        return propagate!(system,system.propagator, δf_res, ρ, cache_propagator...)
    end
end

include("ideal.jl")
include("tangent_hs.jl")