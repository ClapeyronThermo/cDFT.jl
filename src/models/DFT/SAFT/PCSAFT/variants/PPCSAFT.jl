using Clapeyron: PCPSAFTModel, pcp_sigma, pcp_dipole, pcp_dipole2, pcp_epsilon, pcp_segment

"""
    PCPSAFT(components::Vector{String})

The PCP-SAFT equation of state developed by Vrabec and Gross (2006). Our DFT implementation follows the work of Sauer and Gross (2017) which uses a Weighted Density Functional approach and does not use a chain propagator. This uses the same species information as PC-SAFT.

The bulk model can be obtained from Clapeyron. 
"""
PCPSAFT

function f_res(system::DFTSystem, model::PCPSAFTModel,n)
    nd = dimension(system)
    return f_hs(system,model,n[2,:],n[3,:],n[4:4+nd-1,:]) + f_hc(system,model,n[1,:],n[4+nd,:],n[5+nd,:]) + f_disp(system,model,n[6+nd,:]) + f_polar(system,model,n[6+nd,:]) + f_assoc(system,model,n[2,:],n[3,:],n[4:4+nd-1,:])
end

function f_polar(system::DFTSystem, model::PCPSAFTModel, ρ̄)
    species = system.species
    T = system.structure.conditions[2]
    μ̄² = pcp_dipole2(model)
    has_dp = !all(iszero, μ̄²)
    if !has_dp return zero(T+first(ρ̄)) end

    ψ = 1.3862
    HSd = species.size

    m = pcp_segment(model)
    ϵ = pcp_epsilon(model)
    σ = pcp_sigma(model)

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
    η = π/6*@sum(ρ̄[i]*m[i]*HSd[i]^3)
    ∑ρ̄ = sum(ρ̄)
    x = ρ̄ /∑ρ̄
    _A₂ = A2(x,m,ϵ,σ,μ̄²,η,∑ρ̄,T)
    iszero(_A₂) && return zero(_A₂)
    _A₃ = A3(x,m,ϵ,σ,μ̄²,η,∑ρ̄,T)
    _a_dd = _A₂^2/(_A₂-_A₃)
    return ∑ρ̄*_a_dd
end

function A2(x,m,ϵ,σ,μ̄²,η,ρ̄,T)
    p_comps = [i for (i, μ²) ∈ enumerate(μ̄²) if !iszero(μ²)]
    _0 = zero(T+first(x))
    if isempty(p_comps) return _0 end
    _a_2 = _0
    @inbounds for (idx, i) ∈ enumerate(p_comps)
        _J2_ii = J2(m[i],m[i],ϵ[i,i],η,T)
        xᵢ = x[i]
        μ̄²ᵢ = μ̄²[i]
        _a_2 +=xᵢ^2*μ̄²ᵢ^2/σ[i,i]^3*_J2_ii
        for j ∈ p_comps[idx+1:end]
            _J2_ij = J2(m[i],m[j],ϵ[i,j],η,T)
            _a_2 += 2*xᵢ*x[j]*μ̄²ᵢ*μ̄²[j]/σ[i,j]^3*_J2_ij
        end
    end
    _a_2 *= -π*ρ̄/T^2
    return _a_2
end

function A3(x,m,ϵ,σ,μ̄²,η,ρ̄,T)
    p_comps = [i for (i, μ²) ∈ enumerate(μ̄²) if !iszero(μ²)]
    _0 = zero(T+first(x))
    if isempty(p_comps) return _0 end

    _a_3 = _0
    @inbounds for (idx_i,i) ∈ enumerate(p_comps)
        _J3_iii = J3(m[i],m[i],m[i],η)
        xᵢ,μ̄ᵢ² = x[i],μ̄²[i]
        a_3_i = xᵢ*μ̄ᵢ²/σ[i,i]
        _a_3 += a_3_i^3*_J3_iii
        for (idx_j,j) ∈ enumerate(p_comps[idx_i+1:end])
            xⱼ,μ̄ⱼ² = x[j],μ̄²[j]
            σij⁻¹ = 1/σ[i,j]
            a_3_iij = xᵢ*μ̄ᵢ²*σij⁻¹
            a_3_ijj = xⱼ*μ̄ⱼ²*σij⁻¹
            a_3_j = xⱼ*μ̄ⱼ²/σ[j,j]
            _J3_iij = J3(m[i],m[i],m[j],η)
            _J3_ijj = J3(m[i],m[j],m[j],η)
            _a_3 += 3*a_3_iij*a_3_ijj*(a_3_i*_J3_iij + a_3_j*_J3_ijj)
            for k ∈ p_comps[idx_i+idx_j+1:end]
                xₖ,μ̄ₖ² = x[k],μ̄²[k]
                _J3_ijk = J3(m[i],m[j],m[k],η)
                _a_3 += 6*xᵢ*xⱼ*xₖ*μ̄ᵢ²*μ̄ⱼ²*μ̄ₖ²*σij⁻¹/(σ[i,k]*σ[j,k])*_J3_ijk
            end
        end
    end
    _a_3 *= -4*π^2/3*ρ̄^2/T^3
    return _a_3
end

function J2(mᵢ,mⱼ,ϵᵢⱼ,η,T)
    ϵᵢⱼT⁻¹ = ϵᵢⱼ/T
    m̄ = minimum([sqrt(mᵢ*mⱼ), 2.0])

    m1 = 1. - 1/m̄
    m2 = m1 * (1. - 2/m̄)
    corr_a = DD_consts[:corr_a]
    corr_b = DD_consts[:corr_b]

    J_2ij = zero(η)

    for n ∈ 0:4
        a0, a1, a2 = corr_a[n+1]
        b0, b1, b2 = corr_b[n+1]
        a_nij = a0 + a1*m1 + a2*m2
        b_nij = b0 + b1*m1 + b2*m2
        J_2ij += (a_nij + b_nij*ϵᵢⱼT⁻¹) * η^n
    end

    return J_2ij
end

function J3(mᵢ,mⱼ,mₖ,η)
    m̄ = minimum([cbrt(mᵢ*mⱼ*mₖ), 2.0])
    corr_c = DD_consts[:corr_c]
    m1 = 1. - 1/m̄
    m2 = m1 * (1. - 2/m̄)

    J_3ijk = zero(η)
    for n ∈ 0:4
        c0, c1, c2 = corr_c[n+1]
        c_nijk = c0 + c1*m1 + c2*m2
        J_3ijk += c_nijk*η^n
    end

    return J_3ijk
end

const DD_consts = (
    corr_a =
    ((0.3043504,0.9534641,-1.161008),
    (-0.1358588,-1.8396383,4.5258607),
    (1.4493329,2.013118,0.9751222),
    (0.3556977,-7.3724958,-12.281038),
    (-2.0653308,8.2374135,5.9397575)),

    corr_b =
    ((0.2187939,-0.5873164,3.4869576),
    (-1.1896431,1.2489132,-14.915974),
    (1.1626889,-0.508528,15.372022),
    (0.,0.,0.),
    (0.,0.,0.)),

    corr_c =
    ((-0.0646774,-0.9520876,-0.6260979),
    (0.1975882,2.9924258,1.2924686),
    (-0.8087562,-2.3802636,1.6542783),
    (0.6902849,-0.2701261,-3.4396744),
    (0.,0.,0.))
)

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

@inline function _J2_kernel(mᵢ, mⱼ, ϵᵢⱼ, η, T, corr_a, corr_b)
    ϵT = ϵᵢⱼ / T
    m̄  = min(sqrt(mᵢ * mⱼ), 2.0)
    m1  = 1.0 - 1.0/m̄
    m2  = m1 * (1.0 - 2.0/m̄)
    result = 0.0
    ηn = 1.0
    for n in 0:4
        a0, a1, a2 = corr_a[n+1]
        b0, b1, b2 = corr_b[n+1]
        result += (a0 + a1*m1 + a2*m2 + (b0 + b1*m1 + b2*m2)*ϵT) * ηn
        ηn *= η
    end
    return result
end

@inline function _J3_kernel(mᵢ, mⱼ, mₖ, η, corr_c)
    m̄  = min(cbrt(mᵢ * mⱼ * mₖ), 2.0)
    m1  = 1.0 - 1.0/m̄
    m2  = m1 * (1.0 - 2.0/m̄)
    result = 0.0
    ηn = 1.0
    for n in 0:4
        c0, c1, c2 = corr_c[n+1]
        result += (c0 + c1*m1 + c2*m2) * ηn
        ηn *= η
    end
    return result
end

"""
PCP-SAFT dipole–dipole polar term (Padé: A₂²/(A₂−A₃)) at grid point `kk`.
Takes `m̄` and `ηd` from f_disp output.
"""
@inline function f_polar(n, params, T, kk, m̄, ηd, ::Val{NC}, ::Val{ND}) where {NC, ND}
    _pi   = 3.141592653589793
    eps_v = 1e-15
    pcp_m   = params.pcp_m
    pcp_ϵ   = params.pcp_epsilon
    pcp_σ   = params.pcp_sigma
    dip2    = params.dipole2
    ca      = DD_consts.corr_a
    cb      = DD_consts.corr_b
    cc      = DD_consts.corr_c

    has_polar = false
    @inbounds for i in 1:NC
        if dip2[i] != 0.0; has_polar = true; break; end
    end

    res_polar = 0.0
    if has_polar
        ψ       = 1.3862
        idx_ρz  = 6 + ND
        factor  = 3.0 / (4.0*ψ*ψ*ψ*_pi)
        ∑ρ̄_p = eps_v
        @inbounds for i in 1:NC
            ∑ρ̄_p += n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
        end

        _A₂ = 0.0
        @inbounds for i in 1:NC
            dip2_i = dip2[i]
            if dip2_i == 0.0; continue; end
            ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
            xᵢ = ρ̄zi_i / ∑ρ̄_p
            _J2_ii = _J2_kernel(pcp_m[i], pcp_m[i], pcp_ϵ[i,i], ηd, T, ca, cb)
            _A₂ += xᵢ*xᵢ * dip2_i*dip2_i / (pcp_σ[i,i]*pcp_σ[i,i]*pcp_σ[i,i]) * _J2_ii
            @inbounds for j in i+1:NC
                dip2_j = dip2[j]
                if dip2_j == 0.0; continue; end
                ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                xⱼ = ρ̄zi_j / ∑ρ̄_p
                σij3 = pcp_σ[i,j]*pcp_σ[i,j]*pcp_σ[i,j]
                _J2_ij = _J2_kernel(pcp_m[i], pcp_m[j], pcp_ϵ[i,j], ηd, T, ca, cb)
                _A₂ += 2.0 * xᵢ * xⱼ * dip2_i * dip2_j / σij3 * _J2_ij
            end
        end
        _A₂ *= -_pi * ∑ρ̄_p / (T*T)

        if abs(_A₂) > eps_v
            _A₃ = 0.0
            @inbounds for i in 1:NC
                dip2_i = dip2[i]
                if dip2_i == 0.0; continue; end
                ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
                xᵢ = ρ̄zi_i / ∑ρ̄_p
                a3_i = xᵢ * dip2_i / pcp_σ[i,i]
                _J3_iii = _J3_kernel(pcp_m[i], pcp_m[i], pcp_m[i], ηd, cc)
                _A₃ += a3_i*a3_i*a3_i * _J3_iii
                @inbounds for j in i+1:NC
                    dip2_j = dip2[j]
                    if dip2_j == 0.0; continue; end
                    ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                    xⱼ = ρ̄zi_j / ∑ρ̄_p
                    σij⁻¹  = 1.0 / pcp_σ[i,j]
                    a3_iij = xᵢ * dip2_i * σij⁻¹
                    a3_ijj = xⱼ * dip2_j * σij⁻¹
                    a3_j   = xⱼ * dip2_j / pcp_σ[j,j]
                    _J3_iij = _J3_kernel(pcp_m[i], pcp_m[i], pcp_m[j], ηd, cc)
                    _J3_ijj = _J3_kernel(pcp_m[i], pcp_m[j], pcp_m[j], ηd, cc)
                    _A₃ += 3.0 * a3_iij * a3_ijj * (a3_i*_J3_iij + a3_j*_J3_ijj)
                    @inbounds for k in j+1:NC
                        dip2_k = dip2[k]
                        if dip2_k == 0.0; continue; end
                        ρ̄zi_k = n[kk, idx_ρz, k] * factor / (params.HSd[k]*params.HSd[k]*params.HSd[k])
                        xₖ = ρ̄zi_k / ∑ρ̄_p
                        _J3_ijk = _J3_kernel(pcp_m[i], pcp_m[j], pcp_m[k], ηd, cc)
                        _A₃ += 6.0 * xᵢ*xⱼ*xₖ * dip2_i*dip2_j*dip2_k *
                                σij⁻¹ / (pcp_σ[i,k]*pcp_σ[j,k]) * _J3_ijk
                    end
                end
            end
            _A₃ *= -4.0*_pi*_pi/3.0 * ∑ρ̄_p*∑ρ̄_p / (T*T*T)

            denom_p = _A₂ - _A₃
            res_polar = ∑ρ̄_p * _A₂*_A₂ / (denom_p + eps_v)
        end
    end

    return res_polar
end

"""
Pointwise residual free energy for PCP-SAFT: identical to PC-SAFT (HS + HC + disp) with
an additional dipole–dipole polar term (A₂²/(A₂−A₃) Padé approximant).

Field layout (same as PCSAFTModel):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → vector nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc
  5+ND     : ∫ρdz  with d    → λ
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z  (disp + polar)
"""
@inline function f_res(out, n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: PCPSAFTModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc  = f_hc(n, params, T, kk, Val(NC), Val(ND))
    res_disp, m̄, ηd = f_disp(n, params, T, kk, Val(NC), Val(ND))
    res_polar = f_polar(n, params, T, kk, m̄, ηd, Val(NC), Val(ND))
    out[kk] = res_hs + res_hc + res_disp + res_polar
    return nothing
end

function preallocate_params(system::DFTSystem{<:PCPSAFTModel})
    backend = system.options.device
    params = (;
        HSd         = Adapt.adapt(backend, system.species.size),
        m           = Adapt.adapt(backend, system.model.params.segment.values),
        sigma       = Adapt.adapt(backend, system.model.params.sigma.values),
        epsilon     = Adapt.adapt(backend, system.model.params.epsilon.values),
        pcp_m       = Adapt.adapt(backend, pcp_segment(system.model)),
        pcp_sigma   = Adapt.adapt(backend, pcp_sigma(system.model)),
        pcp_epsilon = Adapt.adapt(backend, pcp_epsilon(system.model)),
        dipole2     = Adapt.adapt(backend, pcp_dipole2(system.model)),
    )
    nc = length(system.model)
    return params, nc
end