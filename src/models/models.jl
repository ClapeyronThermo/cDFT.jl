import Clapeyron: a_res

_kernel_type(system::AbstractcDFTSystem) = typeof(system.model)
_kernel_type(system::DGTSystem)          = typeof(system)

include("BasicIdeal.jl")
include("DFT/dft.jl")
include("DGT/dgt.jl")
include("SCFT/scft.jl")

"""
    F_res(system::DFTSystem, ρ)

Residual free energy for DFT systems via the Enzyme/KA kernel path.
Runs the same δf_rev_kernel!/δf_fwd_kernel! used by δFδρ_res! and integrates the primal f_val.
"""
function F_res(system::Union{DFTSystem, DGTSystem}, ρ)
    δfδρ_res, cache_model, _, _ = preallocate(system, ρ)
    δFδρ_res!(system, ρ, δfδρ_res, cache_model...)
    f_val = cache_model[9]
    return ∫(f_val, system.structure)
end

"""
    F_res(system::AbstractcDFTSystem, ρ)

Residual free energy fallback for systems that use the old scalar f_res path (e.g. Electrolyte).
"""
function F_res(system::AbstractcDFTSystem, ρ)
    ngrid  = system.structure.ngrid
    model  = system.model

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
    return ∫(ϕ, system.structure)
end

"""
    δFδρ_res!(system, ρ, δfδρ_res, n, δf, fft_buf, in_buf, out_buf, P, iP,
              params, f_val, δf_val, nc, nd[, fwd_cache])

Enzyme/KernelAbstractions-based functional derivative evaluation. Runs on CPU or GPU
depending on `system.options.device`. When `system.options.ad_mode === :forward`,
`fwd_cache` must be the `(dn_seeds, df_outs, Val(BATCH))` tuple from `preallocate_model`.
"""
function δFδρ_res!(system::AbstractcDFTSystem, ρ, δfδρ_res,
                   n, δf, fft_buf, in_buf, out_buf, P, iP,
                   params, f_val, δf_val, nc, nd, fwd_cache = nothing)
    backend     = system.options.device
    ngrid       = system.structure.ngrid
    NF          = size(n, ndims(n)-1)
    NB          = size(n, ndims(n))
    temperature = convert(fptype(system.options), system.structure.conditions[2])
    M           = _kernel_type(system)

    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)
    synchronize(backend)
    copyto!(n, fft_buf)

    if system.options.ad_mode === :forward
        dn_seeds, df_outs, BATCH_val = fwd_cache
        fill!(δf, 0)
        kernel_fwd_batch = δf_fwd_batch_kernel!(backend)
        kernel_fwd_batch(df_outs, n, f_val, dn_seeds, params, temperature,
                         Val(NF), Val(NB), Val(nc), Val(nd), BATCH_val, M,
                         ndrange = ngrid)
        synchronize(backend)
        for f_idx in 1:NF, c_idx in 1:nc
            k = (f_idx - 1) * nc + c_idx
            selectdim(selectdim(δf, nd+1, f_idx), nd+1, c_idx) .= df_outs[k]
        end
    else  # :reverse (default)
        fill!(δf_val, 1)
        fill!(δf, 0)
        kernel_rev = δf_rev_kernel!(backend)
        kernel_rev(δf, n, f_val, δf_val, params, temperature,
                   Val(NF), Val(NB), Val(nc), Val(nd), M,
                   ndrange = ngrid)
        synchronize(backend)
    end

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
    FP      = fptype(system.options)
    nf      = length_fields(system)
    ngrid   = system.structure.ngrid
    nd      = length(ngrid)
    nb      = size(ρ, nd + 1)

    n       = allocate(backend, FP, ngrid..., nf, nb)
    δf      = allocate(backend, FP, ngrid..., nf, nb)
    fill!(δf, 0)
    fft_buf = allocate(backend, FP, ngrid..., nf, nb)

    CT      = transform_eltype(system.structure, FP)
    in_buf  = allocate(backend, CT, ngrid...)
    out_buf = similar(in_buf)
    tmp     = similar(in_buf)
    plan, iplan = build_transform(system.structure, tmp, nd, backend)

    f_val  = allocate(backend, FP, ngrid...)
    δf_val = allocate(backend, FP, ngrid...)
    fill!(δf_val, 1)

    params, nc = preallocate_params(system)

    if system.options.ad_mode === :forward
        batch = nf * nc
        dn_seeds = ntuple(Val(batch)) do k
            f_idx = (k - 1) ÷ nc + 1
            c_idx = (k - 1) % nc + 1
            seed = allocate(backend, FP, ngrid..., nf, nc)
            fill!(seed, 0)
            fill!(selectdim(selectdim(seed, nd+1, f_idx), nd+1, c_idx), 1)
            seed
        end
        df_outs   = ntuple(_ -> allocate(backend, FP, ngrid...), Val(batch))
        fwd_cache = (dn_seeds, df_outs, Val(batch))
    else
        fwd_cache = nothing
    end

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd, fwd_cache
end

function preallocate_model(system::ElectrolyteDFTSystem, ρ)
    backend = system.options.device
    FP      = fptype(system.options)
    nf      = length_fields(system)
    ngrid   = system.structure.ngrid
    nd      = length(ngrid)
    nb      = size(ρ, nd + 1)

    n       = allocate(backend, FP, ngrid..., nf, nb)
    δf      = allocate(backend, FP, ngrid..., nf, nb)
    fill!(δf, 0)
    fft_buf = allocate(backend, FP, ngrid..., nf, nb)

    CT      = transform_eltype(system.structure, FP)
    in_buf  = allocate(backend, CT, ngrid...)
    out_buf = similar(in_buf)
    tmp     = similar(in_buf)
    plan, iplan = build_transform(system.structure, tmp, nd, backend)

    f_val  = allocate(backend, FP, ngrid...)
    δf_val = allocate(backend, FP, ngrid...)
    fill!(δf_val, 1)

    params, nc = preallocate_params(system)

    # Electrolyte uses its own f_res path; forward-mode batch not yet supported
    if system.options.ad_mode === :forward
        batch = nf * nc
        dn_seeds = ntuple(Val(batch)) do k
            f_idx = (k - 1) ÷ nc + 1
            c_idx = (k - 1) % nc + 1
            seed = allocate(backend, FP, ngrid..., nf, nc)
            fill!(seed, 0)
            fill!(selectdim(selectdim(seed, nd+1, f_idx), nd+1, c_idx), 1)
            seed
        end
        df_outs   = ntuple(_ -> allocate(backend, FP, ngrid...), Val(batch))
        fwd_cache = (dn_seeds, df_outs, Val(batch))
    else
        fwd_cache = nothing
    end

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd, fwd_cache
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
    δf_rev_kernel!(backend)

KernelAbstractions kernel that applies Enzyme reverse-mode AD to `f_res` at each grid point.
Dispatches to the correct `f_res` implementation via `::Type{M}`.
Runs identically on CPU and GPU; the backend is selected at call time.
"""
@kernel function δf_rev_kernel!(
    δf, n, f_val, δf_val, params,
    temperature, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}, ::Type{M}
) where {NF, NB, NC, ND, M}
    kk = @index(Global, Cartesian)
    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Reverse), Const(f_res), Const,
        Const(M), Const(kk),
        Duplicated(f_val, δf_val),
        Duplicated(n, δf),
        Const(params), Const(temperature), Const(Val(NC)), Const(Val(ND))
    )
end

"""
    δf_fwd_kernel!(backend)

KernelAbstractions kernel that applies Enzyme forward-mode AD to `f_res` at each grid point.
Called once per (field, component) direction; `dn` carries the unit-vector seed.
`δf_val[kk]` receives the directional derivative ∂f_res/∂n[kk, f, c].
Uses no per-thread intermediate storage, avoiding GPU stack overflow for large NC.
"""
@kernel function δf_fwd_kernel!(
    δf_val, n, f_val, dn, params,
    temperature, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}, ::Type{M}
) where {NF, NB, NC, ND, M}
    kk = @index(Global, Cartesian)
    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Forward), Const(f_res), Const,
        Const(M), Const(kk),
        Duplicated(f_val, δf_val),
        Duplicated(n, dn),
        Const(params), Const(temperature), Const(Val(NC)), Const(Val(ND))
    )
end

"""
    δf_fwd_batch_kernel!(backend)

Batch forward-mode kernel: computes all NF×NC directional derivatives of `f_res`
in a single Enzyme call using `BatchDuplicated`. Each of the BATCH seeds in `dn_tuple`
has 1.0 at one specific (field, component) position; the corresponding `df_tuple[k][kk]`
receives ∂f_res(n[kk,:])/∂n[kk, f_k, c_k].
"""
@kernel function δf_fwd_batch_kernel!(
    df_tuple, n, f_val, dn_tuple, params,
    temperature, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}, ::Val{BATCH}, ::Type{M}
) where {NF, NB, NC, ND, BATCH, M}
    kk = @index(Global, Cartesian)
    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Forward), Const(f_res), Const,
        Const(M), Const(kk),
        BatchDuplicated(f_val, df_tuple),
        BatchDuplicated(n,     dn_tuple),
        Const(params), Const(temperature), Const(Val(NC)), Const(Val(ND))
    )
end
