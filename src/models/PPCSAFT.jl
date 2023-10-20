function F_res(model::PPCSAFTModel,ρ,T,z)
    ψ = 1.3862
    HSd = d(model,nothing,T,onevec(model))
    dz = ρ[1].mesh_size
    
    _, ρ̄, _ = weights_hs(model,ρ,z,ψ*HSd)

    f(x) = f_polar(model,T,@view(x[@comps]))
    Φ_polar = mapslices(f,ρ̄;dims=2)

    _F_res_PCSAFT = invoke(F_res, Tuple{PCSAFTModel,Any,Any,Any},model,ρ,T,z)
    return _F_res_PCSAFT + ∫(Φ_polar,dz)
end

function δFδρ_res(model::PPCSAFTModel,ρ,T,z)
    _δFδρ_res_PCSAFT = invoke(δFδρ_res, Tuple{PCSAFTModel,Any,Any,Any},model,ρ,T,z)
    return _δFδρ_res_PCSAFT + δFδρ_polar(model,ρ,T,z)
end

function δFδρ_polar(model::PPCSAFTModel,ρ,T,z)
    ψ = 1.3862
    HSd = d(model,nothing,T,onevec(model))
    lim = ψ*HSd

    _, ρ̄, _ = weights_hs(model,ρ,z,lim)

    f(x) = f_polar(model,T,@view(x[@comps]))
    df(x) = ForwardDiff.gradient(f,x)
    
    δfδn0  = mapslices(df,ρ̄;dims=2)
    ∂f∂n0 = δfδn0[:,@comps]

    δFδρ_polar = zeros(length(z),length(model))
    for i in @comps 
        bounds = ρ[i].bounds.+(-lim[i],lim[i])
        ∂f∂n =  DensityProfile(∂f∂n0[:,i],z,bounds,[∂f∂n0[1,i],∂f∂n0[end,i]])
    
        span = range(-lim[i],lim[i],length=101) # Length = 101? Is it because len(z) = 101?

        δFδρ_polar[:,i] = π*∫ρz²dz.(Ref(∂f∂n),z,Ref(span))
    end

    return δFδρ_polar
end

function f_polar(model::PPCSAFTModel,T,ρ̄)
    μ̄² = model.params.dipole2.values
    Q̄² = model.params.quadrupole2.values
    has_dp = !all(iszero, μ̄²)
    has_qp = !all(iszero, Q̄²)
    if !has_dp && !has_qp return zero(T+first(ρ̄)) end

    ψ = 1.3862
    HSd = d(model,nothing,T,onevec(model))
    m = model.params.segment.values
    ϵ = model.params.epsilon.values
    σ = model.params.sigma.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
    η = π/6*sum(ρ̄.*m.*HSd.^3)
    x = ρ̄ /sum(ρ̄)
    x_norm = x ./ sum(x)
    ρ̄ = sum(ρ̄)
    nc = length(model)

    a_mp_total = zero(T+ρ̄)
    a_mp_total += has_dp && a_dd(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    a_mp_total += has_qp && a_qq(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    a_mp_total += has_dp && has_qp && a_dq(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)

    return ρ̄*a_mp_total
end

function a_polar(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
    A₂ = A2(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
    iszero(A₂) && return zero(A₂)
    A₃ = A3(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
    return A₂^2/(A₂-A₃)
end
function a_dd(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    return a_polar(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,:DD)
end
function a_qq(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    return a_polar(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,:QQ)
end
function a_dq(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc)
    return zero(T+first(x_norm))
end

function A2(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
    dp_comps, qp_comps = polar_comps(μ̄²,Q̄²,nc)
    
    P̄² = []
    p_comps = []
    p = 0
    coeff = 0.
    if type == :DD
        if isempty(dp_comps) return 0. end
        P̄² = μ̄²
        p_comps = dp_comps
        p = 3
        coeff = 1.
    end
    if type == :QQ
        if isempty(qp_comps) return 0. end
        P̄² = Q̄²
        p_comps = qp_comps
        p = 7
        coeff = 9/16
    end
    _a_2 = zero(T+first(x_norm))
    @inbounds for (idx, i) ∈ enumerate(p_comps)
        _J2_ii = J2(m[i],m[i],ϵ[i,i],η,T,type)
        zᵢ = x_norm[i]
        P̄²ᵢ = P̄²[i]
        _a_2 +=zᵢ^2*P̄²ᵢ^2/σ[i,i]^p*_J2_ii
        for j ∈ p_comps[idx+1:end]
            _J2_ij = J2(m[i],m[j],ϵ[i,j],η,T,type)
            _a_2 += 2*zᵢ*x_norm[j]*P̄²ᵢ*P̄²[j]/σ[i,j]^p*_J2_ij
        end
    end
    _a_2 *= -π*coeff*ρ̄/T^2
    return _a_2
end

function A3(x_norm,m,ϵ,σ,μ̄²,Q̄²,η,ρ̄,T,nc,type)
    dp_comps, qp_comps = polar_comps(μ̄²,Q̄²,nc)
    P̄² = []
    p_comps = []
    p = 0
    coeff = 0.
    _0 = zero(T+first(x_norm))
    if type == :DD
        if isempty(dp_comps) return _0 end
        P̄² = μ̄²
        p_comps = dp_comps
        p = 1
        coeff = -4*π^2/3
    end
    if type == :QQ
        if isempty(qp_comps) return _0 end
        P̄² = Q̄²
        p_comps = qp_comps
        p = 3
        coeff = 9*π^2/16
    end

    _a_3 = _0
    @inbounds for (idx_i,i) ∈ enumerate(p_comps)
        _J3_iii = J3(m[i],m[i],m[i],η,type)
        zi,P̄²i = x_norm[i],P̄²[i]
        a_3_i = zi*P̄²i/σ[i,i]^p
        _a_3 += a_3_i^3*_J3_iii
        for (idx_j,j) ∈ enumerate(p_comps[idx_i+1:end])
            zj,P̄²j = x_norm[j],P̄²[j]
            σij⁻ᵖ = 1/σ[i,j]^p
            a_3_iij = zi*P̄²i*σij⁻ᵖ
            a_3_ijj = zj*P̄²j*σij⁻ᵖ
            a_3_j = zj*P̄²j/σ[j,j]^p
            _J3_iij = J3(m[i],m[i],m[j],η,type)
            _J3_ijj = J3(m[i],m[j],m[j],η,type)
            _a_3 += 3*a_3_iij*a_3_ijj*(a_3_i*_J3_iij + a_3_j*_J3_ijj)
            for k ∈ p_comps[idx_i+idx_j+1:end]
                zk,P̄²k = x_norm[k],P̄²[k]
                _J3_ijk = J3(m[i],m[j],m[k],η,type)
                _a_3 += 6*zi*zj*zk*P̄²i*P̄²j*P̄²k*σij⁻ᵖ/(σ[i,k]*σ[j,k])^p*_J3_ijk
            end
        end
    end
    _a_3 *= coeff*ρ̄^2/T^3
    return _a_3
end

# TODO
function A3_dq()
    return 0.
end

# Returns [Dipole Comp idxs], [Quadrupole Comp idxs]
function polar_comps(μ̄²,Q̄²,nc)
    dipole_comps = []
    quadrupole_comps = []
    for i in 1:nc
        if !iszero(μ̄²[i]) push!(dipole_comps,i) end
        if !iszero(Q̄²[i]) push!(quadrupole_comps,i) end
    end
    return dipole_comps, quadrupole_comps
end

function J2(mᵢ,mⱼ,ϵᵢⱼ,η,T,type)
    corr_consts = NamedTuple()
    ϵᵢⱼT⁻¹ = ϵᵢⱼ/T
    m_ij = sqrt(mᵢ*mⱼ)
    i_range = 0:4

    if type == :DD
        corr_consts = DD_consts
        m_ij = minimum([m_ij, 2.0])
    elseif type == :DQ
        corr_consts = DQ_consts
        i_range = 0:3 # Needs revision
    elseif type == :QQ
        corr_consts = QQ_consts
    end

    m1 = 1. - 1/m_ij
    m2 = m1 * (1. - 2/m_ij)
    corr_a = corr_consts[:corr_a]
    corr_b = corr_consts[:corr_b]

    J_2ij = 0.

    for n ∈ i_range
        a0n, a1n, a2n = corr_a[n+1]
        b0n, b1n, b2n = corr_b[n+1]
        a_nij = a0n + a1n*m1 + a2n*m2
        b_nij = b0n + b1n*m1 + b2n*m2
        J_2ij += (a_nij + b_nij*ϵᵢⱼT⁻¹) * η^n
    end

    return J_2ij
end

function J3(mᵢ,mⱼ,mₖ,η,type)
    m_ijk = cbrt(mᵢ*mⱼ*mₖ)
    corr_c = ()
    m1 = 1. - 1/m_ijk
    m2 = 0.
    i_range = 0:4

    if type == :DD
        corr_c = DD_consts[:corr_c]
        m_ijk = minimum([m_ijk, 2.0])
        m1 = 1. - 1/m_ijk
        m2 = m1 * (1. - 2/m_ijk)
    elseif type == :DQ
        corr_c = DQ_consts[:corr_c]
        i_range = 0:3
    elseif type == :QQ
        corr_c = QQ_consts[:corr_c]
        m2 = m1 * (1. - 2/m_ijk)
    end

    J_3ijk = 0.
    if type == :DQ
        for n ∈ i_range
            c0n, c1n = corr_c[n+1]
            c_nijk = c0n + c1n*m1
            J_3ijk += c_nijk*η^n
        end
    else
        for n ∈ i_range
            c0n, c1n, c2n = corr_c[n+1]
            c_nijk = c0n + c1n*m1 + c2n*m2
            J_3ijk += c_nijk*η^n
        end
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
    ((7.846431, -20.72202),
    (33.427, -58.63904),
    (4.689111, -1.764887),
    (0., 0.))
)