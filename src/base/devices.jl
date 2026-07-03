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
struct DFTOptions{D, FP<:AbstractFloat}
    device::D
    ad_mode::Symbol   # :reverse (default) or :forward

    function DFTOptions(device::D, ad_mode::Symbol = :forward; precision::Type{FP} = Float64) where {D, FP<:AbstractFloat}
        return new{D,FP}(device, ad_mode)
    end
end

DFTOptions() = DFTOptions(CPU(; static=true))
DFTOptions(device::Backend; ad_mode::Symbol = :forward, precision::Type{FP} = Float64) where FP<:AbstractFloat =
    DFTOptions(device, ad_mode; precision)

fptype(::DFTOptions{D,FP}) where {D,FP} = FP

adapt_to_device(backend, ::Type{FP}, arr::AbstractArray) where FP<:AbstractFloat =
    Adapt.adapt(backend, FP.(arr))

function preallocate(system, ρ)
    backend = system.options.device
    FP = fptype(system.options)

    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    δfδρ_res = allocate(backend, FP, ngrid...,nb)

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
            CT = transform_eltype(system.structure, fptype(system.options))
            Vext = similar(selectdim(ρ, nd+1, 1), CT)
            plan, iplan = build_transform(system.structure, Vext, nd, backend)

            push!(cache_external, (plan, iplan, Vext))
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

export CPU, DFTOptions, fptype