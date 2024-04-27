using Clapeyron: PCPSAFTModel

"""
    PCPSAFT(components::Vector{String})

The PCP-SAFT equation of state developed by Vrabec and Gross (2006). Our DFT implementation follows the work of Sauer and Gross (2017) which uses a Weighted Density Functional approach and does not use a chain propagator. This uses the same species information as PC-SAFT.

The bulk model can be obtained from Clapeyron. 
"""
PCPSAFT

function f_res(system::DFTSystem, model::PCPSAFTModel,n)
    return f_hs(system,model,n[2,:],n[3,:],n[4,:]) + f_hc(system,model,n[1,:],n[5,:],n[6,:]) + f_disp(system,model,n[7,:]) + f_polar(system,model,n[7,:]) + f_assoc(system,model,n[2,:],n[3,:],n[4,:])
end

function f_polar(system::DFTSystem, model::PCPSAFTModel, ρ̄)
    species = system.species
    (_, T, _) = system.structure.conditions
    μ̄² = model.params.dipole2.values
    has_dp = !all(iszero, μ̄²)
    if !has_dp return zero(T+first(ρ̄)) end

    ψ = 1.3862
    HSd = [species[i].size[1] for i in @comps]

    m = model.params.segment.values
    ϵ = model.params.epsilon.values
    σ = model.params.sigma.values

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