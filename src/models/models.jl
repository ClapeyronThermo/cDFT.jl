import Clapeyron: a_res

_kernel_type(system::AbstractcDFTSystem) = typeof(system.model)
_kernel_type(system::DGTSystem)          = typeof(system)

include("BasicIdeal.jl")
include("DFT/dft.jl")
include("DGT/dgt.jl")

"""
    F_res(system::DFTSystem, ρ)

Residual free energy for DFT systems via the Enzyme/KA kernel path.
Runs the same δf_kernel! used by δFδρ_res! and integrates the primal f_val.
"""
function F_res(system::Union{DFTSystem, DGTSystem}, ρ)
    δfδρ_res, cache_model, _, _ = preallocate(system, ρ)
    δFδρ_res!(system, ρ, δfδρ_res, cache_model...)
    f_val = cache_model[9]
    return ∫(f_val, structure_dz(system.structure))
end

"""
    F_res(system::AbstractcDFTSystem, ρ)

Residual free energy fallback for systems that use the old scalar f_res path (e.g. Electrolyte).
"""
function F_res(system::AbstractcDFTSystem, ρ)
    ngrid  = system.structure.ngrid
    model  = system.model
    dz     = structure_dz(system.structure)

    δfδρ_res, cache_model, cache_external, cache_propagator = preallocate(system, ρ)
    n       = cache_model[1]
    fft_buf = cache_model[3]
    in_buf  = cache_model[4]
    out_buf = cache_model[5]
    P       = cache_model[6]
    iP      = cache_model[7]

    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)
    copyto!(n, Adapt.adapt(typeof(n), fft_buf))

    ϕ = similar(ρ, ngrid...)
    ϕ .= 0
    for kk in CartesianIndices(ngrid)
        k = Tuple(kk)
        ϕ[k...] = f_res(system, model, @view(n[k...,:,:]))
    end
    return ∫(ϕ, dz)
end

"""
    δFδρ_res!(system, ρ, δfδρ_res, n, δf, fft_buf, in_buf, out_buf, P, iP,
              params, f_val, δf_val, nc, nd)

Enzyme/KernelAbstractions-based functional derivative evaluation. Runs on CPU or GPU
depending on `system.options.device`.
"""
function δFδρ_res!(system::AbstractcDFTSystem, ρ, δfδρ_res,
                   n, δf, fft_buf, in_buf, out_buf, P, iP,
                   params, f_val, δf_val, nc, nd)
    backend = system.options.device
    ngrid   = system.structure.ngrid
    NF      = size(n, ndims(n)-1)
    NB      = size(n, ndims(n))
    T       = system.structure.conditions[2]

    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)
    synchronize(backend)
    copyto!(n, fft_buf)

    fill!(δf_val, 1.0)
    fill!(δf, 0.0)

    kernel = δf_kernel!(backend)
    kernel(δf, n, f_val, δf_val, params, Float64(T),
           Val(NF), Val(NB), Val(nc), Val(nd), _kernel_type(system),
           ndrange = ngrid)
    synchronize(backend)

    copyto!(fft_buf, δf)
    integrate_field!(system, fft_buf, δfδρ_res, in_buf, P, iP)
end

"""
    δFδρ_res(system::DFTSystem, ρ)

Obtain the functional derivatives of the residual free energy for each component / bead.
Output is a 2D array `(ngrid, nb)`, normalised by `kB*T`.
"""
function δFδρ_res(system::AbstractcDFTSystem, ρ)
    δfδρ_res, cache_model, cache_external, cache_propagator = preallocate(system, ρ)
    δFδρ_res!(system, ρ, δfδρ_res, cache_model...)
    evaluate_external_field!(system, ρ, δfδρ_res, cache_external)
    propagate!(system, ρ, δfδρ_res, cache_propagator)
    return δfδρ_res
end

"""
    preallocate_model(system::DFTSystem, ρ)

Generic Enzyme/KernelAbstractions preallocator for all DFT models.
Allocates shared buffers, then delegates model-specific parameter assembly
to `preallocate_params(system)` which returns `(params, nc)`.
"""
function preallocate_model(system::DFTSystem, ρ)
    backend = system.options.device
    nf      = length_fields(system)
    ngrid   = system.structure.ngrid
    nd      = length(ngrid)
    nb      = size(ρ, nd + 1)

    n       = allocate(backend, Float64, ngrid..., nf, nb)
    δf      = allocate(backend, Float64, ngrid..., nf, nb)
    fill!(δf, 0.0)
    fft_buf = allocate(backend, Float64, ngrid..., nf, nb)

    in_buf  = allocate(backend, ComplexF64, ngrid...)
    out_buf = similar(in_buf)
    tmp     = similar(in_buf)
    if backend isa CPU
        plan = plan_fft!(tmp, 1:nd; num_threads = Threads.nthreads())
    else
        plan = plan_fft!(tmp, 1:nd)
    end
    iplan = inv(plan)

    f_val  = allocate(backend, Float64, ngrid...)
    δf_val = allocate(backend, Float64, ngrid...)
    fill!(δf_val, 1.0)

    params, nc = preallocate_params(system)

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd
end

function preallocate_model(system::ElectrolyteDFTSystem, ρ)
    backend = system.options.device
    nf      = length_fields(system)
    ngrid   = system.structure.ngrid
    nd      = length(ngrid)
    nb      = size(ρ, nd + 1)

    n       = allocate(backend, Float64, ngrid..., nf, nb)
    δf      = allocate(backend, Float64, ngrid..., nf, nb)
    fill!(δf, 0.0)
    fft_buf = allocate(backend, Float64, ngrid..., nf, nb)

    in_buf  = allocate(backend, ComplexF64, ngrid...)
    out_buf = similar(in_buf)
    tmp     = similar(in_buf)
    if backend isa CPU
        plan = plan_fft!(tmp, 1:nd; num_threads = Threads.nthreads())
    else
        plan = plan_fft!(tmp, 1:nd)
    end
    iplan = inv(plan)

    f_val  = allocate(backend, Float64, ngrid...)
    δf_val = allocate(backend, Float64, ngrid...)
    fill!(δf_val, 1.0)

    params, nc = preallocate_params(system)

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd
end

function length_scales(model::EoSModel)
    if hasfield(typeof(model.params), :sigma)
        return diagvalues(model.params.sigma.values)
    elseif hasfield(typeof(model.params), :b)
        return diagvalues(cbrt.(model.params.b.values / N_A))
    elseif hasfield(typeof(model.params), :lb_volume)
        return cbrt.(model.params.lb_volume.values / N_A)
    else
        error("No length scale defined in model")
    end
end

"""
    δf_kernel!(backend)

KernelAbstractions kernel that applies Enzyme reverse-mode AD to `f_res` at each grid point.
Dispatches to the correct `f_res` implementation via `::Type{M}`.
Runs identically on CPU and GPU; the backend is selected at call time.
"""
@kernel function δf_kernel!(
    δf, n, f_val, δf_val, params,
    T, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}, ::Type{M}
) where {NF, NB, NC, ND, M}
    kk = @index(Global, Cartesian)
    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Reverse), Const(f_res), Const,
        Duplicated(f_val, δf_val),
        Duplicated(n, δf),
        Const(params), Const(Float64(T)), Const(kk),
        Const(Val(NC)), Const(Val(ND)), Const(M)
    )
end
