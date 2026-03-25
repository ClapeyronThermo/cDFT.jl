import KernelAbstractions: Backend, get_backend, synchronize
using Base: ScopedValue

# Default float precision per backend.
# Metal does not support Float64, so GPU backends default to Float32.
# Users can override via the `precision` keyword of DFTOptions.
_default_precision(::CPU)     = Float64
_default_precision(::Backend) = Float32

"""
    DFTOptions(device, solver; precision)

Settings for convergence algorithms and the compute device.

# Arguments
- `device`: CPU or GPU backend (unpinned CPU by default).
- `solver`: Fixed-point solver (`AndersonFixPoint` by default).

# Keyword Arguments
- `precision`: Element type for device arrays (`Float32` or `Float64`).
  Defaults to `Float64` on CPU and `Float32` on GPU backends (Metal does not
  support Float64). CUDA and ROCm support both; pass `precision=Float64` to
  opt into double precision on those backends.

# Examples
```julia
options = DFTOptions()                                    # CPU, Float64
options = DFTOptions(CUDABackend())                       # CUDA, Float32
options = DFTOptions(CUDABackend(); precision=Float64)    # CUDA, Float64
options = DFTOptions(MetalBackend())                      # Metal, Float32
```
"""
struct DFTOptions{D, S, T<:AbstractFloat}
    device    :: D
    solver    :: S
    precision :: Type{T}
end

function DFTOptions(device::Backend, solver; precision::Type{<:AbstractFloat}=_default_precision(device))
    return DFTOptions(device, solver, precision)
end

function DFTOptions(device::Backend; precision::Type{<:AbstractFloat}=_default_precision(device))
    return DFTOptions(device, AndersonFixPoint(), precision)
end

function DFTOptions(; precision::Type{<:AbstractFloat}=Float64)
    return DFTOptions(CPU(; static=true), AndersonFixPoint(), precision)
end

"""
    float_type(options::DFTOptions) -> Type

Return the floating-point scalar type used for device allocations.
Controlled by the `precision` field of `DFTOptions`.
"""
float_type(opts::DFTOptions) = opts.precision

is_metal_backend(backend) = nameof(typeof(backend)) == :MetalBackend

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

function preallocate_model(system, ρ)
    backend = system.options.device
    
    nf = length_fields(system)
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)
    n = allocate(CPU(), Float64, ngrid...,nf,nb)
    δf = allocate(CPU(), Float64, ngrid...,nf,nb)

    fft_buf = allocate(backend, Float64, ngrid...,nf,nb)

    in_buf = allocate(backend, ComplexF64, ngrid...)
    out_buf = similar(in_buf)              #

    tmp = similar(in_buf)
    if backend isa CPU
        plan = plan_fft!(tmp, 1:length(ngrid); num_threads=Threads.nthreads())
    else
        plan = plan_fft!(tmp, 1:length(ngrid))
    end

    iplan = inv(plan)

    f(x) = f_res(system,system.model,x)
    idx_first = ntuple(Returns(1),nd)
    n_first = @view(n[idx_first...,:,:])

    chunksize = ForwardDiff.Chunk(system)

    first_config = ForwardDiff.GradientConfig(f, n_first, chunksize)

    cache_pool = Channel{typeof(first_config)}(Threads.nthreads())
    for _ in 1:Threads.nthreads()
        put!(cache_pool, ForwardDiff.GradientConfig(f, n_first, chunksize))
    end
    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, f, cache_pool
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
