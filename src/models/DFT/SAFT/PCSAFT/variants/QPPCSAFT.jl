using Clapeyron: QPCPSAFTModel, pcp_sigma, pcp_dipole2, pcp_epsilon, pcp_segment

const QQ_consts = (
    corr_a = 
    ((1.2378308, 1.2854109,	1.7942954),
    (2.4355031,	-11.465615,	0.7695103),
    (1.6330905,	22.086893,	7.2647923),
    (-1.6118152, 7.4691383,	94.486699),
    (6.9771185,	-17.197772,	-77.148458)),

    corr_b = 
    ((0.4542718, -0.813734, 6.8682675),
    (-4.5016264, 10.06403, -5.1732238),
    (3.5858868,	-10.876631, -17.240207),
    (0., 0., 0.),
    (0., 0., 0.)),

    corr_c =
    ((-0.5000437, 2.0002094, 3.1358271),
    (6.5318692, -6.7838658, 7.2475888),
    (-16.01478, 20.383246, 3.0759478),
    (14.42597, -10.895984, 0.),
    (0., 0., 0.))    
)

const DQ_consts = (
    corr_a =
    ((0.697095, -0.6734593, 0.6703408),
    (-0.6335541, -1.4258991, -4.3384718),
    (2.945509, 4.1944139, 7.2341684),
    (-1.4670273, 1.0266216, 0.)),

    corr_b =
    ((-0.4840383, 0.6765101, -1.1675601),
    (1.9704055, -3.0138675, 2.1348843),
    (-2.1185727, 0.4674266, 0.),
    (0.,	0.,	0.)),

    corr_c =
    ((0.7846431, -2.072202),
    (3.3427, -5.863904),
    (0.4689111, -0.1764887),
    (0., 0.))
)

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
QQ J₂ integral kernel: no min(m̄,2) clamping (unlike DD), 5 terms.
"""
@inline function _J2_qq_kernel(mᵢ, mⱼ, ϵᵢⱼ, η, T, corr_a, corr_b)
    ϵT = ϵᵢⱼ / T
    m̄  = sqrt(mᵢ * mⱼ)
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

"""
DQ J₂ integral kernel: no min clamping, 4 terms only (DQ_consts has 4 entries).
"""
@inline function _J2_dq_kernel(mᵢ, mⱼ, ϵᵢⱼ, η, T, corr_a, corr_b)
    ϵT = ϵᵢⱼ / T
    m̄  = sqrt(mᵢ * mⱼ)
    m1  = 1.0 - 1.0/m̄
    m2  = m1 * (1.0 - 2.0/m̄)
    result = 0.0
    ηn = 1.0
    for n in 0:3
        a0, a1, a2 = corr_a[n+1]
        b0, b1, b2 = corr_b[n+1]
        result += (a0 + a1*m1 + a2*m2 + (b0 + b1*m1 + b2*m2)*ϵT) * ηn
        ηn *= η
    end
    return result
end

"""
QQ J₃ integral kernel: no min clamping, 5 terms, includes m2.
"""
@inline function _J3_qq_kernel(mᵢ, mⱼ, mₖ, η, corr_c)
    m̄  = cbrt(mᵢ * mⱼ * mₖ)
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
DQ J₃ integral kernel: no min clamping, 4 terms, no m2 (DQ_consts.corr_c has only (c0,c1) per entry).
"""
@inline function _J3_dq_kernel(mᵢ, mⱼ, mₖ, η, corr_c)
    m̄  = cbrt(mᵢ * mⱼ * mₖ)
    m1  = 1.0 - 1.0/m̄
    result = 0.0
    ηn = 1.0
    for n in 0:3
        c0, c1 = corr_c[n+1]
        result += (c0 + c1*m1) * ηn
        ηn *= η
    end
    return result
end

"""
QPCP-SAFT polar term at grid point `kk`: Padé sum of DD + QQ + DQ contributions.
"""
@inline function f_polar(n, params, T, kk, m̄, ηd, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: QPCPSAFTModel}
    _pi   = 3.141592653589793
    eps_v = 1e-15
    pcp_m = params.pcp_m
    pcp_ϵ = params.pcp_epsilon
    pcp_σ = params.pcp_sigma
    dip2  = params.dipole2
    quad2 = params.quadrupole2
    ca_dd = DD_consts.corr_a;  cb_dd = DD_consts.corr_b;  cc_dd = DD_consts.corr_c
    ca_qq = QQ_consts.corr_a;  cb_qq = QQ_consts.corr_b;  cc_qq = QQ_consts.corr_c
    ca_dq = DQ_consts.corr_a;  cb_dq = DQ_consts.corr_b;  cc_dq = DQ_consts.corr_c

    ψ      = 1.3862
    idx_ρz = 6 + ND
    factor = 3.0 / (4.0*ψ*ψ*ψ*_pi)
    ∑ρ̄_p  = eps_v
    @inbounds for i in 1:NC
        ∑ρ̄_p += n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
    end

    has_dipole = false;  has_quad = false
    @inbounds for i in 1:NC
        if dip2[i]  != 0.0; has_dipole = true; end
        if quad2[i] != 0.0; has_quad   = true; end
    end

    res_polar = 0.0

    # ── DD contribution ───────────────────────────────────────────────────────
    if has_dipole
        _A₂_dd = 0.0
        @inbounds for i in 1:NC
            dip2_i = dip2[i]
            if dip2_i == 0.0; continue; end
            ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
            xᵢ = ρ̄zi_i / ∑ρ̄_p
            σii3 = pcp_σ[i,i]*pcp_σ[i,i]*pcp_σ[i,i]
            _J2_ii = _J2_kernel(pcp_m[i], pcp_m[i], pcp_ϵ[i,i], ηd, T, ca_dd, cb_dd)
            _A₂_dd += xᵢ*xᵢ * dip2_i*dip2_i / σii3 * _J2_ii
            @inbounds for j in i+1:NC
                dip2_j = dip2[j]
                if dip2_j == 0.0; continue; end
                ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                xⱼ = ρ̄zi_j / ∑ρ̄_p
                σij3 = pcp_σ[i,j]*pcp_σ[i,j]*pcp_σ[i,j]
                _J2_ij = _J2_kernel(pcp_m[i], pcp_m[j], pcp_ϵ[i,j], ηd, T, ca_dd, cb_dd)
                _A₂_dd += 2.0 * xᵢ * xⱼ * dip2_i * dip2_j / σij3 * _J2_ij
            end
        end
        _A₂_dd *= -_pi * ∑ρ̄_p / (T*T)

        if abs(_A₂_dd) > eps_v
            _A₃_dd = 0.0
            @inbounds for i in 1:NC
                dip2_i = dip2[i]
                if dip2_i == 0.0; continue; end
                ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
                xᵢ = ρ̄zi_i / ∑ρ̄_p
                a3_i = xᵢ * dip2_i / pcp_σ[i,i]
                _J3_iii = _J3_kernel(pcp_m[i], pcp_m[i], pcp_m[i], ηd, cc_dd)
                _A₃_dd += a3_i*a3_i*a3_i * _J3_iii
                @inbounds for j in i+1:NC
                    dip2_j = dip2[j]
                    if dip2_j == 0.0; continue; end
                    ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                    xⱼ = ρ̄zi_j / ∑ρ̄_p
                    σij⁻¹  = 1.0 / pcp_σ[i,j]
                    a3_iij = xᵢ * dip2_i * σij⁻¹
                    a3_ijj = xⱼ * dip2_j * σij⁻¹
                    a3_j   = xⱼ * dip2_j / pcp_σ[j,j]
                    _J3_iij = _J3_kernel(pcp_m[i], pcp_m[i], pcp_m[j], ηd, cc_dd)
                    _J3_ijj = _J3_kernel(pcp_m[i], pcp_m[j], pcp_m[j], ηd, cc_dd)
                    _A₃_dd += 3.0 * a3_iij * a3_ijj * (a3_i*_J3_iij + a3_j*_J3_ijj)
                    @inbounds for k in j+1:NC
                        dip2_k = dip2[k]
                        if dip2_k == 0.0; continue; end
                        ρ̄zi_k = n[kk, idx_ρz, k] * factor / (params.HSd[k]*params.HSd[k]*params.HSd[k])
                        xₖ = ρ̄zi_k / ∑ρ̄_p
                        _J3_ijk = _J3_kernel(pcp_m[i], pcp_m[j], pcp_m[k], ηd, cc_dd)
                        _A₃_dd += 6.0 * xᵢ*xⱼ*xₖ * dip2_i*dip2_j*dip2_k *
                                    σij⁻¹ / (pcp_σ[i,k]*pcp_σ[j,k]) * _J3_ijk
                    end
                end
            end
            _A₃_dd *= -4.0*_pi*_pi/3.0 * ∑ρ̄_p*∑ρ̄_p / (T*T*T)
            denom_dd = _A₂_dd - _A₃_dd
            res_polar += ∑ρ̄_p * _A₂_dd*_A₂_dd / (denom_dd + eps_v)
        end
    end

    # ── QQ contribution ───────────────────────────────────────────────────────
    if has_quad
        _A₂_qq = 0.0
        @inbounds for i in 1:NC
            quad2_i = quad2[i]
            if quad2_i == 0.0; continue; end
            ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
            xᵢ = ρ̄zi_i / ∑ρ̄_p
            σii_sq = pcp_σ[i,i]*pcp_σ[i,i]
            σii7   = σii_sq*σii_sq*σii_sq*pcp_σ[i,i]
            _J2_ii = _J2_qq_kernel(pcp_m[i], pcp_m[i], pcp_ϵ[i,i], ηd, T, ca_qq, cb_qq)
            _A₂_qq += xᵢ*xᵢ * quad2_i*quad2_i / σii7 * _J2_ii
            @inbounds for j in i+1:NC
                quad2_j = quad2[j]
                if quad2_j == 0.0; continue; end
                ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                xⱼ = ρ̄zi_j / ∑ρ̄_p
                σij_sq = pcp_σ[i,j]*pcp_σ[i,j]
                σij7   = σij_sq*σij_sq*σij_sq*pcp_σ[i,j]
                _J2_ij = _J2_qq_kernel(pcp_m[i], pcp_m[j], pcp_ϵ[i,j], ηd, T, ca_qq, cb_qq)
                _A₂_qq += 2.0 * xᵢ * xⱼ * quad2_i * quad2_j / σij7 * _J2_ij
            end
        end
        _A₂_qq *= -(9.0/16.0) * _pi * ∑ρ̄_p / (T*T)

        if abs(_A₂_qq) > eps_v
            _A₃_qq = 0.0
            @inbounds for i in 1:NC
                quad2_i = quad2[i]
                if quad2_i == 0.0; continue; end
                ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
                xᵢ = ρ̄zi_i / ∑ρ̄_p
                σii3 = pcp_σ[i,i]*pcp_σ[i,i]*pcp_σ[i,i]
                a3_i = xᵢ * quad2_i / σii3
                _J3_iii = _J3_qq_kernel(pcp_m[i], pcp_m[i], pcp_m[i], ηd, cc_qq)
                _A₃_qq += a3_i*a3_i*a3_i * _J3_iii
                @inbounds for j in i+1:NC
                    quad2_j = quad2[j]
                    if quad2_j == 0.0; continue; end
                    ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                    xⱼ = ρ̄zi_j / ∑ρ̄_p
                    σjj3   = pcp_σ[j,j]*pcp_σ[j,j]*pcp_σ[j,j]
                    σij3   = pcp_σ[i,j]*pcp_σ[i,j]*pcp_σ[i,j]
                    a3_iij = xᵢ * quad2_i / σij3
                    a3_ijj = xⱼ * quad2_j / σij3
                    a3_j   = xⱼ * quad2_j / σjj3
                    _J3_iij = _J3_qq_kernel(pcp_m[i], pcp_m[i], pcp_m[j], ηd, cc_qq)
                    _J3_ijj = _J3_qq_kernel(pcp_m[i], pcp_m[j], pcp_m[j], ηd, cc_qq)
                    _A₃_qq += 3.0 * a3_iij * a3_ijj * (a3_i*_J3_iij + a3_j*_J3_ijj)
                    @inbounds for k in j+1:NC
                        quad2_k = quad2[k]
                        if quad2_k == 0.0; continue; end
                        ρ̄zi_k = n[kk, idx_ρz, k] * factor / (params.HSd[k]*params.HSd[k]*params.HSd[k])
                        xₖ = ρ̄zi_k / ∑ρ̄_p
                        σik3 = pcp_σ[i,k]*pcp_σ[i,k]*pcp_σ[i,k]
                        σjk3 = pcp_σ[j,k]*pcp_σ[j,k]*pcp_σ[j,k]
                        _J3_ijk = _J3_qq_kernel(pcp_m[i], pcp_m[j], pcp_m[k], ηd, cc_qq)
                        _A₃_qq += 6.0 * xᵢ*xⱼ*xₖ * quad2_i*quad2_j*quad2_k /
                                    (σij3*σik3*σjk3) * _J3_ijk
                    end
                end
            end
            _A₃_qq *= (9.0*_pi*_pi/16.0) * ∑ρ̄_p*∑ρ̄_p / (T*T*T)
            denom_qq = _A₂_qq - _A₃_qq
            res_polar += ∑ρ̄_p * _A₂_qq*_A₂_qq / (denom_qq + eps_v)
        end
    end

    # ── DQ cross contribution ─────────────────────────────────────────────────
    if has_dipole && has_quad
        _A₂_dq = 0.0
        @inbounds for i in 1:NC
            dip2_i = dip2[i]
            if dip2_i == 0.0; continue; end
            ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
            xᵢ = ρ̄zi_i / ∑ρ̄_p
            @inbounds for j in 1:NC
                quad2_j = quad2[j]
                if quad2_j == 0.0; continue; end
                ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                xⱼ = ρ̄zi_j / ∑ρ̄_p
                σij_sq = pcp_σ[i,j]*pcp_σ[i,j]
                σij5   = σij_sq*σij_sq*pcp_σ[i,j]
                _J2_ij = _J2_dq_kernel(pcp_m[i], pcp_m[j], pcp_ϵ[i,j], ηd, T, ca_dq, cb_dq)
                _A₂_dq += xᵢ * xⱼ * dip2_i * quad2_j / σij5 * _J2_ij
            end
        end
        _A₂_dq *= -(9.0/4.0) * _pi * ∑ρ̄_p / (T*T)

        if abs(_A₂_dq) > eps_v
            _A₃_dq = 0.0
            @inbounds for i in 1:NC
                dip2_i = dip2[i]
                if dip2_i == 0.0; continue; end
                ρ̄zi_i = n[kk, idx_ρz, i] * factor / (params.HSd[i]*params.HSd[i]*params.HSd[i])
                xᵢ = ρ̄zi_i / ∑ρ̄_p
                @inbounds for j in 1:NC
                    # j ∈ dp_comps ∪ qp_comps: contributes if it has dipole or quadrupole
                    if dip2[j] == 0.0 && quad2[j] == 0.0; continue; end
                    ρ̄zi_j = n[kk, idx_ρz, j] * factor / (params.HSd[j]*params.HSd[j]*params.HSd[j])
                    xⱼ = ρ̄zi_j / ∑ρ̄_p
                    contrib_j = pcp_σ[j,j] * dip2[j] + 1.19374 / pcp_σ[j,j] * quad2[j]
                    @inbounds for k in 1:NC
                        quad2_k = quad2[k]
                        if quad2_k == 0.0; continue; end
                        ρ̄zi_k = n[kk, idx_ρz, k] * factor / (params.HSd[k]*params.HSd[k]*params.HSd[k])
                        xₖ = ρ̄zi_k / ∑ρ̄_p
                        σij2 = pcp_σ[i,j]*pcp_σ[i,j]
                        σik2 = pcp_σ[i,k]*pcp_σ[i,k]
                        σjk2 = pcp_σ[j,k]*pcp_σ[j,k]
                        _J3_ijk = _J3_dq_kernel(pcp_m[i], pcp_m[j], pcp_m[k], ηd, cc_dq)
                        _A₃_dq += xᵢ * xⱼ * xₖ * pcp_σ[i,i] /
                                    (pcp_σ[k,k] * σij2*σik2*σjk2) *
                                    dip2_i * quad2_k * contrib_j * _J3_ijk
                    end
                end
            end
            _A₃_dq *= -∑ρ̄_p*∑ρ̄_p / (T*T*T)
            denom_dq = _A₂_dq - _A₃_dq
            res_polar += ∑ρ̄_p * _A₂_dq*_A₂_dq / (denom_dq + eps_v)
        end
    end

    return res_polar
end

"""
Pointwise residual free energy for QPCP-SAFT: HS + HC + disp + polar (DD + QQ + DQ).

Field layout identical to PCP-SAFT / PC-SAFT:
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → vector nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc
  5+ND     : ∫ρdz  with d    → λ
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z  (disp + polar)
"""
@inline function f_res(out, n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: QPCPSAFTModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc  = f_hc(n, params, T, kk, Val(NC), Val(ND), M)
    res_disp, m̄, ηd = f_disp(n, params, T, kk, Val(NC), Val(ND), M)
    res_polar = f_polar(n, params, T, kk, m̄, ηd, Val(NC), Val(ND), M)
    res_assoc = _assoc_or_zero(n, params, T, kk, Val(NC), Val(ND), M)
    out[kk] = res_hs + res_hc + res_disp + res_polar + res_assoc
    return nothing
end

function preallocate_params(system::DFTSystem{<:QPCPSAFTModel})
    backend = system.options.device
    model   = system.model
    base = (;
        HSd         = Adapt.adapt(backend, system.species.size),
        m           = Adapt.adapt(backend, model.params.segment.values),
        sigma       = Adapt.adapt(backend, model.params.sigma.values),
        epsilon     = Adapt.adapt(backend, model.params.epsilon.values),
        pcp_m       = Adapt.adapt(backend, pcp_segment(model)),
        pcp_sigma   = Adapt.adapt(backend, pcp_sigma(model)),
        pcp_epsilon = Adapt.adapt(backend, pcp_epsilon(model)),
        dipole2     = Adapt.adapt(backend, pcp_dipole2(model)),
        quadrupole2 = Adapt.adapt(backend, model.params.quadrupole2.values),
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params(model, system.species.size)

        nc_model = length(model)
        assoc_icomp_t    = ntuple(p -> p <= nn           ? assoc_icomp_v[p]    : 0, Val(20))
        assoc_jcomp_t    = ntuple(p -> p <= nn           ? assoc_jcomp_v[p]    : 0, Val(20))
        assoc_isite_t    = ntuple(p -> p <= nn           ? assoc_isite_v[p]    : 0, Val(20))
        assoc_jsite_t    = ntuple(p -> p <= nn           ? assoc_jsite_v[p]    : 0, Val(20))
        n_sites_flat_t   = ntuple(j -> j <= total_sites  ? n_sites_flat_v[j]   : 0, Val(20))
        n_sites_cumsum_t = ntuple(i -> i <= nc_model + 1 ? n_sites_cumsum_v[i] : 0, Val(11))

        assoc = (;
            has_assoc      = true,
            assoc_n_pairs  = Val(nn),
            assoc_icomp    = assoc_icomp_t,
            assoc_jcomp    = assoc_jcomp_t,
            assoc_isite    = assoc_isite_t,
            assoc_jsite    = assoc_jsite_t,
            assoc_eps      = Adapt.adapt(backend, assoc_eps_v),
            assoc_kap      = Adapt.adapt(backend, assoc_kap_v),
            assoc_sig3     = Adapt.adapt(backend, assoc_sig3_v),
            assoc_dij      = Adapt.adapt(backend, assoc_dij_v),
            n_sites_flat   = n_sites_flat_t,
            n_sites_cumsum = n_sites_cumsum_t,
            total_sites,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    return params, length(model)
end