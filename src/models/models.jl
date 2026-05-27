import Clapeyron: a_res

include("BasicIdeal.jl")
include("DFT/dft.jl")
include("DGT/dgt.jl")

# Constants for PCSAFT dispersion (Matching Clapeyron exactly)
const PCSAFT_CORR1 = (
    (0.9105631445, -0.3084016918, -0.0906148351),
    (0.6361281449, 0.1860531159, 0.4527842806),
    (2.6861347891, -2.5030047259, 0.5962700728),
    (-26.547362491, 21.419793629, -1.7241829131),
    (97.759208784, -65.25588533, -4.1302112531),
    (-159.59154087, 83.318680481, 13.77663187),
    (91.297774084, -33.74692293, -8.6728470368)
)
const PCSAFT_CORR2 = (
    (0.7240946941, -0.5755498075, 0.0976883116),
    (2.2382791861, 0.6995095521, -0.2557574982),
    (-4.0025849485, 3.892567339, -9.155856153),
    (-21.003576815, -17.215471648, 20.642075974),
    (26.855641363, 192.67226447, -38.804430052),
    (206.55133841, -161.82646165, 93.626774077),
    (-355.60235612, -165.20769346, -29.666905585)
)

@inline function I_lite(corr, m̄, η)
    res = 0.0
    m2 = (m̄ - 1.0) / m̄
    m3 = m2 * (m̄ - 2.0) / m̄
    # Manual unrolling for GPU safety and AD clarity
    c1 = corr[1]; res += (c1[1] + m2 * c1[2] + m3 * c1[3])
    c2 = corr[2]; res += (c2[1] + m2 * c2[2] + m3 * c2[3]) * η
    c3 = corr[3]; res += (c3[1] + m2 * c3[2] + m3 * c3[3]) * η * η
    c4 = corr[4]; res += (c4[1] + m2 * c4[2] + m3 * c4[3]) * η * η * η
    c5 = corr[5]; res += (c5[1] + m2 * c5[2] + m3 * c5[3]) * η * η * η * η
    c6 = corr[6]; res += (c6[1] + m2 * c6[2] + m3 * c6[3]) * η * η * η * η * η
    c7 = corr[7]; res += (c7[1] + m2 * c7[2] + m3 * c7[3]) * η * η * η * η * η * η
    return res
end

@inline function f_res_lite_void_gpu(out, n, HSd, m, σ, ϵ, T, kk, ::Val{NC}, ::Val{ND}) where {NC, ND}
    _pi = 3.141592653589793
    eps_val = 1e-15
    
    # --- f_hs logic (Matching FMT.jl) ---
    n₀ = 0.0; n₁ = 0.0; n₂ = 0.0; n₃₃ = 0.0
    nv1_1 = 0.0; nv1_2 = 0.0; nv1_3 = 0.0
    nv2_1 = 0.0; nv2_2 = 0.0; nv2_3 = 0.0
    
    @inbounds for i in 1:NC
        mi = m[i]; HSdi = HSd[i]
        ni_2 = n[kk, 2, i] # Field 2: weight 0.5*d
        nim = ni_2 * mi
        n₀ += nim / HSdi
        n₁ += 0.5 * nim
        n₂ += _pi * nim * HSdi
        
        # Vector fields (Field 4)
        if ND >= 1; nvi = n[kk, 4, i]; nv1_1 += nvi * mi / HSdi; nv2_1 += -2.0 * _pi * nvi * mi; end
        if ND >= 2; nvi = n[kk, 5, i]; nv1_2 += nvi * mi / HSdi; nv2_2 += -2.0 * _pi * nvi * mi; end
        if ND >= 3; nvi = n[kk, 6, i]; nv1_3 += nvi * mi / HSdi; nv2_3 += -2.0 * _pi * nvi * mi; end
        
        # Field 3: weight 0.5*d (type :∫ρz²dz -> volume fraction n3)
        n₃₃ += n[kk, 3, i] * mi
    end
    
    n_v1_dot_v2 = nv1_1 * nv2_1 + nv1_2 * nv2_2 + nv1_3 * nv2_3
    n_v2_sq = nv2_1 * nv2_1 + nv2_2 * nv2_2 + nv2_3 * nv2_3
    
    denom_log = 1.0 - n₃₃
    log_1_n33 = Base.log(abs(denom_log) + eps_val)
    inv_1_n₃₃ = 1.0 / (denom_log + eps_val)
    
    denom_f_hs = 12.0 * _pi * (n₃₃ * n₃₃ + eps_val)
    bracket_term = log_1_n33 / denom_f_hs + 1.0 / (12.0 * _pi * (n₃₃ + eps_val) * (denom_log * denom_log + eps_val))
    res_f_hs = -n₀ * log_1_n33 + (n₁ * n₂ - n_v1_dot_v2) * inv_1_n₃₃ + 
           (n₂ * n₂ * n₂ / 3.0 - n₂ * n_v2_sq) * bracket_term

    # --- f_hc logic (Matching PCSAFT.jl) ---
    ζ₃ = 0.0; ζ₂ = 0.0
    idx_ζ = 4 + ND # Field 5: weight d (type :∫ρz²dz)
    @inbounds for i in 1:NC
        mi = m[i]; ρ̄hci = n[kk, idx_ζ, i]; HSdi = HSd[i]
        ζ₃ += mi * ρ̄hci; ζ₂ += mi * ρ̄hci / HSdi
    end
    ζ₃ *= 0.125; ζ₂ *= 0.125
    inv_1_ζ₃ = 1.0 / (1.0 - ζ₃ + eps_val)
    
    res_f_hc = 0.0
    idx_λ = 5 + ND # Field 6: weight d (type :∫ρdz)
    @inbounds for i in 1:NC
        ρi = n[kk, 1, i]
        λ = n[kk, idx_λ, i] / (2.0 * HSd[i])
        yᵈᵈ = inv_1_ζ₃ + 1.5 * HSd[i] * ζ₂ * inv_1_ζ₃ * inv_1_ζ₃ + 0.5 * HSd[i] * HSd[i] * ζ₂ * ζ₂ * inv_1_ζ₃ * inv_1_ζ₃ * inv_1_ζ₃
        res_f_hc += -ρi * (m[i] - 1.0) * Base.log(abs(yᵈᵈ * λ / (ρi + eps_val)) + eps_val)
    end

    # --- f_disp logic (Matching PCSAFT.jl) ---
    ψ = 1.3862
    ρ̄z_sum = eps_val; m̄_top = 0.0; η_disp = 0.0
    factor = 3.0 / (4.0 * ψ * ψ * ψ * _pi)
    idx_ρz = 6 + ND # Field 7: weight d*ψ
    @inbounds for i in 1:NC
        ρ̄zi = n[kk, idx_ρz, i] * factor / (HSd[i] * HSd[i] * HSd[i])
        ρ̄z_sum += ρ̄zi
        m̄_top += ρ̄zi * m[i]
        η_disp += m[i] * ρ̄zi * HSd[i] * HSd[i] * HSd[i]
    end
    m̄ = m̄_top / ρ̄z_sum
    ηd = _pi / 6.0 * η_disp
    
    m2ϵσ3_1 = 0.0; m2ϵσ3_2 = 0.0
    @inbounds for i in 1:NC
        ρzi = n[kk, idx_ρz, i] * factor / (HSd[i] * HSd[i] * HSd[i])
        @inbounds for j in i:NC
            ρzj = n[kk, idx_ρz, j] * factor / (HSd[j] * HSd[j] * HSd[j])
            const_ij = ρzi * ρzj * m[i] * m[j] * σ[i,j] * σ[i,j] * σ[i,j]
            eps_T = ϵ[i,j] / (T + eps_val)
            term1 = const_ij * eps_T
            term2 = const_ij * eps_T * eps_T
            if i == j
                m2ϵσ3_1 += term1
                m2ϵσ3_2 += term2
            else
                m2ϵσ3_1 += 2.0 * term1
                m2ϵσ3_2 += 2.0 * term2
            end
        end
    end
    
    ηd2 = ηd * ηd; ηd4 = (1.0 - ηd + eps_val)^4
    inv_1_ηd = 1.0 / (1.0 - ηd + eps_val)
    inv_2_ηd = 1.0 / (2.0 - ηd + eps_val)
    C₁ = 1.0 + m̄ * (8.0 * ηd - 2.0 * ηd2) / ηd4 + (1.0 - m̄) * (20.0 * ηd - 27.0 * ηd2 + 12.0 * (ηd * ηd2) - 2.0 * (ηd2 * ηd2)) * inv_1_ηd * inv_1_ηd * inv_2_ηd * inv_2_ηd
    I₁ = I_lite(PCSAFT_CORR1, m̄, ηd)
    I₂ = I_lite(PCSAFT_CORR2, m̄, ηd)
    
    res_f_disp = -2.0 * _pi * I₁ * m2ϵσ3_1 - _pi * m̄ * I₂ * m2ϵσ3_2 / (C₁ + eps_val)
    
    out[kk] = res_f_hs + res_f_hc + res_f_disp
    return nothing
end

"""
    F_res(system::DFTSystem, ρ)

Obtain the residual free energy of the system for a given profile `ρ`. This is done by first evaluating the system fields and passing these to the integrands (`f_res`) for each grid point. The result is then integrated over the domain.

The output is a scalar of units J.
"""
function F_res(system::AbstractcDFTSystem, ρ)
    ngrid = system.structure.ngrid
    bounds = system.structure.bounds
    model = system.model

    _bounds = system.structure.bounds

    δfδρ_res, cache_model, cache_external, cache_propagator = preallocate(system, ρ)
    (n, δf, fft_buf, in_buf, out_buf, P, iP, f, cache_pool) = cache_model
    dz = structure_dz(system.structure)
    
    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)
    
    copyto!(n, Adapt.adapt(typeof(n), fft_buf))

    ϕ = similar(ρ,ngrid...)
    ϕ .= 0
    for kk in CartesianIndices(ngrid)
        k = Tuple(kk)
        ϕ[k...] = f(@view(n[k...,:,:]))
    end

    return ∫(ϕ,dz)
end

"""
    δFδρ_res(system::DFTSystem, ρ)

Obtain the functional derivatives of the residual free energy of the system for each component / bead for a given profile `ρ`. This is done by first evaluating the system fields, obtaining the derivative of the integrands (`f_res`) for each grid point and then integrating over each of these fields to obtain the functional derivatives.

The output is a 2D array with the dimensions `(ngrid,nb)`, where `ngrid` is the number of grid points, and `nb` is the number of beads in the model. The values are normalised by `kB*T`.
"""

function δFδρ_res!(system::AbstractcDFTSystem, ρ, δfδρ_res, n, δf, fft_buf, in_buf, out_buf, P, iP, f, cache_pool)
    model = system.model
    backend = system.options.device
    ngrid = system.structure.ngrid
    NF      = size(n, ndims(n)-1)
    NB      = size(n, ndims(n))
    ND      = length(ngrid)
    # println("backend = ", backend)

    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)

    synchronize(backend)

    copyto!(n, Adapt.adapt(typeof(n), fft_buf))

    Threads.@threads for kk in CartesianIndices(ngrid)
        k = Tuple(kk)
        cache = take!(cache_pool)
        ForwardDiff.gradient!(@view(δf[k...,:,:]), f, @view(n[k...,:,:]), cache)
        put!(cache_pool, cache)
    end

    copyto!(fft_buf, Adapt.adapt(typeof(fft_buf), δf))

    synchronize(backend)


    integrate_field!(system, fft_buf, δfδρ_res, in_buf, P, iP)
end

function δFδρ_res(system::AbstractcDFTSystem, ρ)
    δfδρ_res, cache_model, cache_external, cache_propagator = preallocate(system, ρ)
    δFδρ_res!(system, ρ, δfδρ_res, cache_model...)
    evaluate_external_field!(system, ρ, δfδρ_res, cache_external)
    propagate!(system, ρ, δfδρ_res, cache_propagator)
    return δfδρ_res
end

function length_scales(model::EoSModel)
    if hasfield(typeof(model.params), :sigma)
        return diagvalues(model.params.sigma.values)
    elseif hasfield(typeof(model.params), :b)
        return diagvalues(cbrt.(model.params.b.values/N_A))
    elseif hasfield(typeof(model.params), :lb_volume)
        return cbrt.(model.params.lb_volume.values/N_A)
    else
        error("No length scale defined in model")
    end
end

function δFδρ_res_newautodiff!(system::AbstractcDFTSystem, ρ, δfδρ_res, n, δf, fft_buf, in_buf, out_buf, P, iP, f, cache_pool, f_val, δf_val, HSd=nothing, m=nothing, sigma=nothing, epsilon=nothing, T=nothing, nc=nothing, nd=nothing)
    model = system.model
    backend = system.options.device
    # println("Running new autodiff version with backend: ", backend, "\n")
    

    # Fallback to original version if association is present or on CPU or no flat data
    if Clapeyron.assoc_pair_length(model) > 0  || isnothing(HSd)
        return δFδρ_res!(system, ρ, δfδρ_res, n, δf, fft_buf, in_buf, out_buf, P, iP, f, cache_pool)
    end

    # GPU-accelerated version using Enzyme
    ngrid = system.structure.ngrid
    NF = size(n, ndims(n)-1)
    NB = size(n, ndims(n))

    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)
    synchronize(backend)

    # Copy fields to n buffer
    copyto!(n, fft_buf)
    
    # Initialize gradient output buffer to zero as Enzyme accumulates
    # fill!(δf, 0.0)
    # synchronize(backend)
    # Launch Enzyme autodiff kernel with raw arrays and static NC, ND
    kernel = δf_enzyme_kernel!(backend)
    kernel(δf, n, f_val, δf_val, HSd, m, sigma, epsilon, Float64(T), 
           Val(NF), Val(NB), Val(nc), Val(nd),
           ndrange=ngrid)
    synchronize(backend)

    # Copy gradients back to fft_buf for integration
    copyto!(fft_buf, δf)

    integrate_field!(system, fft_buf, δfδρ_res, in_buf, P, iP)
end

function δFδρ_res_newautodiff(system::AbstractcDFTSystem, ρ)
    backend = system.options.device
    # Fallback for CPU or models with association
    if Clapeyron.assoc_pair_length(system.model) > 0
        return δFδρ_res(system, ρ)
    end

    δfδρ_res, cache_model, cache_external, cache_propagator = preallocate_newautodiff(system, ρ)
    δFδρ_res_newautodiff!(system, ρ, δfδρ_res, cache_model...)
    evaluate_external_field!(system, ρ, δfδρ_res, cache_external)
    propagate!(system, ρ, δfδρ_res, cache_propagator)
    return δfδρ_res
end

function preallocate_newautodiff(system, ρ)
    backend = system.options.device
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    δfδρ_res = allocate(backend, Float64, ngrid...,nb)
    cache_model = preallocate_model_newautodiff(system, ρ)
    cache_external = preallocate_external_potential(system, ρ)
    cache_propagator = preallocate_propagator(system, ρ)

    return δfδρ_res, cache_model, cache_external, cache_propagator
end

function preallocate_model_newautodiff(system, ρ)
    backend = system.options.device
    # if backend isa CPU
    #     return preallocate_model(system, ρ)
    # end

    # GPU-specific allocation
    nf = length_fields(system)
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)
    
    n = allocate(backend, Float64, ngrid...,nf,nb)
    δf = allocate(backend, Float64, ngrid...,nf,nb)
    fill!(δf, 0.0)
    fft_buf = allocate(backend, Float64, ngrid...,nf,nb)

    in_buf = allocate(backend, ComplexF64, ngrid...)
    out_buf = similar(in_buf)

    tmp = similar(in_buf)
    plan = plan_fft!(tmp, 1:length(ngrid))
    iplan = inv(plan)

    # Local function value buffers for Enzyme Global Void pattern
    f_val = allocate(backend, Float64, ngrid...)
    δf_val = allocate(backend, Float64, ngrid...)
    fill!(δf_val, 1.0)

    # Decompose model for GPU bitstype safety
    if system.model isa PCSAFTModel
        HSd = Adapt.adapt(backend, system.species.size)
        m = Adapt.adapt(backend, system.model.params.segment.values)
        sigma = Adapt.adapt(backend, system.model.params.sigma.values)
        epsilon = Adapt.adapt(backend, system.model.params.epsilon.values)
        T = system.structure.conditions[2]
        nc = length(system.model)
        _nd = dimension(system)
    else
        HSd = m = sigma = epsilon = T = nc = _nd = nothing
    end
    
    # Standard function slot for fallback
    f = x -> f_res(system, system.model, x)

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, f, nothing, f_val, δf_val, HSd, m, sigma, epsilon, T, nc, _nd
end

@kernel function δf_enzyme_kernel!(
    δf, n, f_val, δf_val, HSd, m, sigma, epsilon, 
    T, ::Val{NF}, ::Val{NB}, ::Val{NC}, ::Val{ND}
) where {NF, NB, NC, ND}
    kk = @index(Global, Cartesian)
    
    # Enzyme Reverse AD for local integrand gradient using Global Void pattern
    # We differentiate f_res_lite_void_gpu wrt global memory n
    # Val(NC) and Val(ND) are passed as Const to enforce active inference.
    Enzyme.autodiff_deferred(Reverse, Const(f_res_lite_void_gpu), Const, 
        Duplicated(f_val, δf_val),
        Duplicated(n, δf), 
        Const(HSd), Const(m), Const(sigma), Const(epsilon), 
        Const(Float64(T)), Const(kk), Const(Val(NC)), Const(Val(ND)))
end