import Clapeyron: COFFEEModel, Iμμ, ∫∫∫Odξ₁dξ₂dγ12, ∫∫∫dξ₁dξ₂dγ12, ∫odr, COFFEEconsts
import Clapeyron: pcp_segment, pcp_sigma, pcp_epsilon, pcp_dipole2

"""
    COFFEE(components::Vector{String})

The COFFEE equation of state developed by Langenbach (2017). This is an unpublished approach which uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron. 
"""
COFFEE

function get_fields(model::COFFEEModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    nc = length(model)
    ψ = 1.3862

    ω = structure_ω(structure, device, FP)
    d = species.size
    ngrid = structure.ngrid
    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = species.size ./ σ
    ψ1 = @. cbrt(3*C*x^3*(x^-λ_a/(λ_a-3)-x^-λ_r/(λ_r-3)))
    return [SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,ψ*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,ψ1.*d,ω,ngrid,device)]
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: COFFEEModel}
    res_hs, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3 =
        f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(1))
    res_disp = f_disp(M, kk, n, params, T, Val(NC), Val(ND), Val(4+ND))
    res_ff = f_ff(M, kk, n, params, T, Val(NC), Val(ND))
    res_nf = f_nf(M, kk, n, params, T, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3, Val(NC), Val(ND))
    out[kk] = res_hs + res_disp + res_ff + res_nf
    return nothing
end

"""
Pointwise residual free energy for COFFEE:
FMT hard-sphere + SAFT-VR Mie dispersion + far-field polar (f_ff) + near-field polar (f_nf).
No chain (IdealPropagator).

Field layout (5 fields):
  1        : ∫ρdz  with 0.5*d → n₀, n₁, n₂ (FMT + f_nf)
  2        : ∫ρz²dz with 0.5*d → n₃ (FMT + f_nf η)
  3..2+ND  : ∫ρzdz with 0.5*d → nᵥ (FMT + f_nf)
  3+ND     : ∫ρz²dz with ψ*d (ψ=1.3862) → far-field polar density
  4+ND     : ∫ρz²dz with ψ1*d → dispersion density (per-component ψ1)
"""
@inline function f_ff(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: COFFEEModel}
    HSd   = params.HSd
    m_seg = params.m
    pcp_ϵ = params.pcp_epsilon
    pcp_σ = params.pcp_sigma
    dip2  = params.dipole2
    ψff   = 1.3862
    idx_ff = 3 + ND
    cb2   = params.coffee_b2;  cc2 = params.coffee_c2
    cb3   = params.coffee_b3;  cc3 = params.coffee_c3

    ∑ρ̄_ff = 0.0;  η_ff = 0.0
    @inbounds for i in 1:NC
        di    = HSd[i]
        ρ̄i_ff = n[kk,idx_ff,i] * 3.0/(4.0*ψff^3*π*di*di*di)
        ∑ρ̄_ff += ρ̄i_ff
        η_ff  += m_seg[i] * ρ̄i_ff * di*di*di
    end
    η_ff *= π/6.0

    has_polar = false
    @inbounds for i in 1:NC
        if dip2[i] != 0.0; has_polar = true; break; end
    end

    res_polar = 0.0
    if has_polar
        _A₂ = 0.0
        @inbounds for i in 1:NC
            d2i = dip2[i]; if d2i == 0.0; continue; end
            ρ̄i = n[kk,idx_ff,i] * 3.0/(4.0*ψff^3*π*HSd[i]^3)
            xi = ρ̄i / ∑ρ̄_ff
            J2ii = _J2_coffee_kernel(pcp_ϵ[i,i], η_ff, T, cb2, cc2)
            _A₂ += xi*xi * d2i*d2i / (pcp_σ[i,i]*pcp_σ[i,i]*pcp_σ[i,i]) * J2ii
            @inbounds for j in (i+1):NC
                d2j = dip2[j]; if d2j == 0.0; continue; end
                ρ̄j = n[kk,idx_ff,j] * 3.0/(4.0*ψff^3*π*HSd[j]^3)
                xj = ρ̄j / ∑ρ̄_ff
                σij3 = pcp_σ[i,j]*pcp_σ[i,j]*pcp_σ[i,j]
                J2ij = _J2_coffee_kernel(pcp_ϵ[i,j], η_ff, T, cb2, cc2)
                _A₂ += 2.0*xi*xj * d2i*d2j / σij3 * J2ij
            end
        end
        _A₂ *= -π * ∑ρ̄_ff / (T*T)

        if abs(_A₂) > 0.0
            _A₃ = 0.0
            @inbounds for i in 1:NC
                d2i = dip2[i]; if d2i == 0.0; continue; end
                ρ̄i = n[kk,idx_ff,i] * 3.0/(4.0*ψff^3*π*HSd[i]^3)
                xi  = ρ̄i / ∑ρ̄_ff
                a3i = xi * d2i / pcp_σ[i,i]
                J3iii = _J3_coffee_kernel(pcp_ϵ[i,i], η_ff, T, cb3, cc3)
                _A₃ += a3i*a3i*a3i * J3iii
                @inbounds for j in (i+1):NC
                    d2j = dip2[j]; if d2j == 0.0; continue; end
                    ρ̄j = n[kk,idx_ff,j] * 3.0/(4.0*ψff^3*π*HSd[j]^3)
                    xj   = ρ̄j / ∑ρ̄_ff
                    σij⁻¹ = 1.0 / pcp_σ[i,j]
                    a3iij = xi*d2i*σij⁻¹;  a3ijj = xj*d2j*σij⁻¹
                    a3j   = xj*d2j/pcp_σ[j,j]
                    J3iij = _J3_coffee_kernel(pcp_ϵ[i,j], η_ff, T, cb3, cc3)
                    J3ijj = _J3_coffee_kernel(pcp_ϵ[i,j], η_ff, T, cb3, cc3)
                    _A₃ += 3.0*a3iij*a3ijj*(a3i*J3iij + a3j*J3ijj)
                    @inbounds for kk2 in (j+1):NC
                        d2k = dip2[kk2]; if d2k == 0.0; continue; end
                        ρ̄k = n[kk,idx_ff,kk2] * 3.0/(4.0*ψff^3*π*HSd[kk2]^3)
                        xk  = ρ̄k / ∑ρ̄_ff
                        J3ijk = _J3_coffee_kernel(pcp_ϵ[i,j], η_ff, T, cb3, cc3)
                        _A₃ += 6.0*xi*xj*xk * d2i*d2j*d2k *
                                σij⁻¹/(pcp_σ[i,kk2]*pcp_σ[j,kk2]) * J3ijk
                    end
                end
            end
            _A₃ *= -4.0*π*π/3.0 * ∑ρ̄_ff*∑ρ̄_ff / (T*T*T)
            denom_p = _A₂ - _A₃
            res_polar = ∑ρ̄_ff * _A₂*_A₂ / denom_p
        end
    end
    return res_polar
end

# GPU-safe inline helpers for COFFEE polar terms

@inline function _J2_coffee_kernel(ϵij, η, T, b2, c2)
    eT = ϵij / T
    c0 = b2[1] + c2[1]*eT;  c1 = b2[2] + c2[2]*eT
    c2v= b2[3] + c2[3]*eT;  c3 = b2[4] + c2[4]*eT
    c4 = b2[5] + c2[5]*eT
    return evalpoly(η, (c0, c1, c2v, c3, c4))
end

@inline function _J3_coffee_kernel(ϵij, η, T, b3, c3)
    eT = ϵij / T
    c0 = b3[1] + c3[1]*eT;  c1 = b3[2] + c3[2]*eT
    c2v= b3[3] + c3[3]*eT;  c3v= b3[4] + c3[4]*eT
    return evalpoly(η, (c0, c1, c2v, c3v))
end

@inline function f_nf(::Type{M}, kk, n, params, T, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: COFFEEModel}
    HSd       = params.HSd
    nf_d      = params.nf_d
    nf_mu2    = params.nf_mu2
    nf_mu     = sqrt(abs(nf_mu2))
    nf_eps    = params.nf_epsilon
    nf_sigrat = params.nf_sigma_ratio
    corr_I    = params.coffee_corr_I
    xi_x      = params.xi_x;  xi_w = params.xi_w
    gamma_x   = params.gamma_x;  gamma_w = params.gamma_w

    ρ̄_nf   = n₃₃ * 6.0/π * nf_sigrat
    T̄_nf   = T / nf_eps
    Iμμval = _Iμμ_kernel(ρ̄_nf, T̄_nf, nf_mu, corr_I)

    coeff = -24.0/19.0 * nf_mu2 / T̄_nf * Iμμval
    Q_nf = 0.0
    @inbounds for i in 1:25
        ξ1i = xi_x[i]; w1i = xi_w[i]
        @inbounds for j in 1:25
            ξ2j = xi_x[j]; w2j = xi_w[j]
            @inbounds for kk3 in 1:20
                γk  = gamma_x[kk3]; wγk = gamma_w[kk3]
                Iv  = _∫odr_kernel(nf_d, ξ1i, ξ2j, γk)
                Q_nf += w1i * w2j * wγk * exp(coeff * Iv)
            end
        end
    end

    nv2sq_nf = nv2_1*nv2_1 + nv2_2*nv2_2 + nv2_3*nv2_3
    di1  = HSd[1]
    ξ_nf = 1.0 - nv2sq_nf / (n₂*n₂)
    g_hs_nf = 1.0/(1.0-n₃₃) + di1*ξ_nf*n₂/(4.0*(1.0-n₃₃)^2) +
              di1*di1/4.0*n₂*n₂*ξ_nf/(18.0*(1.0-n₃₃)^3)

    return 19.0*π/12.0 * n₀ * ρ̄_nf * g_hs_nf * Base.log(4.0*π/Q_nf)
end

@inline function _Iμμ_kernel(ρ̄, T̄, μ, a)
    return a[1] + a[2]/T̄ + a[3]/(T̄*T̄) + ρ̄*(
        a[4] + a[5]*μ + a[6]/T̄ + a[7]/(T̄*T̄) + a[8]*ρ̄ + a[9]*ρ̄*ρ̄ + a[10]*ρ̄*μ/T̄)
end

@inline function _∫odr_kernel(d, ξ1, ξ2, γ12)
    if d == 0.0
        s1 = sqrt(abs(1.0 - ξ1*ξ1));  s2 = sqrt(abs(1.0 - ξ2*ξ2))
        cosΘ = 2.0*ξ1*ξ2 - s1*s2*cos(γ12)
        return -cosΘ * 5.0/18.0
    else
        s1 = sqrt(abs(1.0 - ξ1*ξ1));  s2 = sqrt(abs(1.0 - ξ2*ξ2))
        A12 = ξ1*ξ2 + s1*s2*cos(γ12)
        a_  = 3.0*ξ1*ξ2 - A12
        b_  = d*(ξ1 - ξ2)*(A12 - 3.0)
        c_  = -d*d*(A12*A12 - 4.0*A12 + 3.0)
        f_  = 2.0*d*(ξ1 - ξ2)
        f2  = f_*f_;  f3 = f2*f_
        g_  = -2.0*d*d*(A12 - 1.0)
        g2  = g_*g_
        inv_3f24g2 = 1.0 / (3.0*(f2 - 4.0*g_)*(f2 - 4.0*g_))
        term1 = (1.0/((1.0+f_+g_)^1.5)) * (
                2.0*c_*(2.0+f_)*(-8.0-8.0*f_+f2-12.0*g_)
              + 2.0*b_*(3.0*f3+8.0*g2+2.0*f2*(6.0+g_)+4.0*f_*(2.0+3.0*g_))
              - 2.0*a_*(3.0*f3+8.0*g_+4.0*f_*g_*(3.0+2.0*g_)+2.0*f2*(1.0+6.0*g_)))
        term2 = (1.0/((9.0+6.0*f_+4.0*g_)^1.5)) * 4.0 * (
              - 4.0*c_*(3.0+f_)*(-12.0*f_+f2-6.0*(3.0+2.0*g_))
              - 2.0*b_*(9.0*f3+16.0*g2+18.0*f_*(3.0+2.0*g_)+f2*(54.0+4.0*g_))
              + a_*(27.0*f3+108.0*g_+9.0*f2*(3.0+8.0*g_)+4.0*f_*g_*(27.0+8.0*g_)))
        return -inv_3f24g2 * (term1 + term2)
    end
end

function preallocate_params(system::DFTSystem{<:COFFEEModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model
    σ_diag  = diagvalues(model.params.sigma.values)
    ϵ_diag  = diagvalues(model.params.epsilon.values)
    nf_mu2_val = pcp_dipole2(model)[1] / ϵ_diag[1] / σ_diag[1]^3
    nf_d_val   = model.params.shift[1] / σ_diag[1]
    nf_sig3    = (σ_diag[1] / system.species.size[1])^3

    nc = length(model)
    lr = model.params.lambda_r.values
    la = model.params.lambda_a.values
    lambda_r_t = ntuple(i -> ntuple(j -> lr[i,j], nc), nc)
    lambda_a_t = ntuple(i -> ntuple(j -> la[i,j], nc), nc)
    params = (;
        HSd         = adapt_to_device(backend, FP, system.species.size),
        m           = adapt_to_device(backend, FP, model.params.segment.values),
        meff        = adapt_to_device(backend, FP, model.params.segment.values),
        sigma       = adapt_to_device(backend, FP, model.params.sigma.values),
        epsilon     = adapt_to_device(backend, FP, model.params.epsilon.values),
        lambda_r_t  = lambda_r_t,
        lambda_a_t  = lambda_a_t,
        psi_eff     = adapt_to_device(backend, FP, system.fields[end].width),
        A           = SAFTVRMIE_A,
        phi         = SAFTVRMIE_PHI,
        pcp_m       = adapt_to_device(backend, FP, pcp_segment(model)),
        pcp_sigma   = adapt_to_device(backend, FP, pcp_sigma(model)),
        pcp_epsilon = adapt_to_device(backend, FP, pcp_epsilon(model)),
        dipole2     = adapt_to_device(backend, FP, pcp_dipole2(model)),
        coffee_b2   = COFFEEconsts.corr_b2,
        coffee_c2   = COFFEEconsts.corr_c2,
        coffee_b3   = COFFEEconsts.corr_b3,
        coffee_c3   = COFFEEconsts.corr_c3,
        coffee_corr_I = COFFEEconsts.corr_I,
        nf_d        = nf_d_val,
        nf_mu2      = nf_mu2_val,
        nf_epsilon  = ϵ_diag[1],
        nf_sigma_ratio = nf_sig3,
        xi_x        = COFFEEconsts.xi_quadrature[1],
        xi_w        = COFFEEconsts.xi_quadrature[2],
        gamma_x     = COFFEEconsts.gamma_quadrature[1],
        gamma_w     = COFFEEconsts.gamma_quadrature[2],
    )
    nc = length(model)
    return params, nc
end