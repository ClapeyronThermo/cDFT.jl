abstract type Device end

"""
    DFTOptions(device::Device, solver::Solvers.AbstractFixPoint)

A struct which includes all the settings that need to be set for the convergence algorithms and devices used:
- `device`: Specification of either CPU (pinned or un-pinned) or GPU devices. (unpinned CPU by default)
- `solver`: Specification of the solver type and solver settings used. Must be a fixed-point method. (`AndersonFixPoint` by default)
Example usage:
```julia
julia> options = DFTOptions()

julia> using ThreadPinning

julia> options = DFTOptions(CPU(4, [0,1,12,13]))
```
"""
struct DFTCache
    n::Array{Float64}
end

function DFTCache(ngrid,nfields,nbeads)
    return DFTCache(zeros(ngrid,nfields,nbeads))
end

export CPU, GPU, DFTOptions