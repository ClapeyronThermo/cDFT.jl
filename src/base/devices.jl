import KernelAbstractions: Backend, get_backend, synchronize
using Base: ScopedValue
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
struct DFTOptions{D}
    device::D
end

# function DFTOptions(device::Backend)
#     return DFTOptions(device)
# end

function DFTOptions()
    return DFTOptions(CPU(; static=true))
end

function preallocate(system, ρ)
    backend = system.options.device
    
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    δfδρ_res = allocate(backend, Float64, ngrid...,nb)

    cache_model = preallocate_model(system, ρ)

    cache_external = preallocate_external_potential(system, ρ)

    cache_propagator = preallocate_propagator(system, ρ)

    return δfδρ_res, cache_model, cache_external, cache_propagator
end


function preallocate_external_potential(system, ρ)
    backend = system.options.device
    ngrid = system.structure.ngrid
    nd = length(ngrid)

    cache_external = Any[]

    external_fields = system.external_field
    if isnothing(external_fields)
        return nothing
    end
    for external_field in external_fields
    
        if external_field isa ElectrostaticPotentialModel
            Vext = similar(selectdim(ρ, nd+1, 1), ComplexF64)

            if backend isa CPU
                plan = plan_fft!(Vext, 1:length(ngrid); num_threads=Threads.nthreads())
            else
                plan = plan_fft!(Vext, 1:length(ngrid))
            end

            
            push!(cache_external, (plan, inv(plan), Vext))
        else
            # Vext = allocate(backend, Float64, system.structure.ngrid...)

            z = get_coords(system.structure)

            Vext = Adapt.adapt(typeof(ρ),evaluate_external_field!(system.structure, external_field, system.model, ρ, ρ, z))

            push!(cache_external, (z,Vext))
        end
    end
    
    return cache_external
end

function preallocate_propagator(system, ρ)
    backend = system.options.device
    propagtor = system.propagator

    return preallocate_propagator(system, propagtor, ρ, backend)
end


export CPU, DFTOptions