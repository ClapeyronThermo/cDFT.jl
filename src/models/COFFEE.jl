import Clapeyron: COFFEEModel, Iμμ, ∫∫∫Odξ₁dξ₂dγ12, ∫∫∫dξ₁dξ₂dγ12, ∫odr, COFFEEconsts

"""
    COFFEE(components::Vector{String})

The COFFEE equation of state developed by Langenbach (2017). This is an unpublished approach which uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron. 
"""
COFFEE

function get_fields(model::COFFEEModel, species::DFTSpecies, structure::DFTStructure)
    nc = length(model)
    ψ = 1.3862

    f = structure.ngrid/(structure.bounds[2]-structure.bounds[1])
    ω = fftfreq(structure.ngrid, f)
    d = species.size

    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = species.size ./ σ
    ψ1 = @. cbrt(3*C*x^3*(x^-λ_a/(λ_a-3)-x^-λ_r/(λ_r-3)))
    return [WeightedDensity(:∫ρdz,0.5*d,ω),
            WeightedDensity(:∫ρz²dz,0.5*d,ω),
            WeightedDensity(:∫ρzdz,0.5*d,ω),
            WeightedDensity(:∫ρz²dz,ψ*d,ω),
            WeightedDensity(:∫ρz²dz,ψ1.*d,ω)]
end

function f_res(system::DFTSystem, model::COFFEEModel,n)
    return f_hs(system,model,n[1,:],n[2,:],n[3,:])+f_disp(system,model,n[5,:])+f_ff(system,model,n[4,:])+f_nf(system,model,n[1,:],n[2,:],n[3,:])
end

function f_ff(system::DFTSystem, model::COFFEEModel, ρ̄)
    species = system.species
    (_, T) = system.structure.conditions
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
    _A₂ = A2_coffee(x,m,ϵ,σ,μ̄²,η,∑ρ̄,T)
    iszero(_A₂) && return zero(_A₂)
    _A₃ = A3_coffee(x,m,ϵ,σ,μ̄²,η,∑ρ̄,T)
    _a_dd = _A₂^2/(_A₂-_A₃)
    return ∑ρ̄*_a_dd
end

function A2_coffee(x,m,ϵ,σ,μ̄²,η,ρ̄,T)
    p_comps = [i for (i, μ²) ∈ enumerate(μ̄²) if !iszero(μ²)]
    _0 = zero(T+first(x))
    if isempty(p_comps) return _0 end
    _a_2 = _0
    @inbounds for (idx, i) ∈ enumerate(p_comps)
        _J2_ii = J2_coffee(T,i,i,η,ϵ)
        xᵢ = x[i]
        μ̄²ᵢ = μ̄²[i]
        _a_2 +=xᵢ^2*μ̄²ᵢ^2/σ[i,i]^3*_J2_ii
        for j ∈ p_comps[idx+1:end]
            _J2_ij = J2_coffee(T,i,j,η,ϵ)
            _a_2 += 2*xᵢ*x[j]*μ̄²ᵢ*μ̄²[j]/σ[i,j]^3*_J2_ij
        end
    end
    _a_2 *= -π*ρ̄/T^2
    return _a_2
end

function A3_coffee(x,m,ϵ,σ,μ̄²,η,ρ̄,T)
    p_comps = [i for (i, μ²) ∈ enumerate(μ̄²) if !iszero(μ²)]
    _0 = zero(T+first(x))
    if isempty(p_comps) return _0 end

    _a_3 = _0
    @inbounds for (idx_i,i) ∈ enumerate(p_comps)
        _J3_iii = J3_coffee(T,i,i,i,η,ϵ)
        xᵢ,μ̄ᵢ² = x[i],μ̄²[i]
        a_3_i = xᵢ*μ̄ᵢ²/σ[i,i]
        _a_3 += a_3_i^3*_J3_iii
        for (idx_j,j) ∈ enumerate(p_comps[idx_i+1:end])
            xⱼ,μ̄ⱼ² = x[j],μ̄²[j]
            σij⁻¹ = 1/σ[i,j]
            a_3_iij = xᵢ*μ̄ᵢ²*σij⁻¹
            a_3_ijj = xⱼ*μ̄ⱼ²*σij⁻¹
            a_3_j = xⱼ*μ̄ⱼ²/σ[j,j]
            _J3_iij = J3_coffee(T,i,i,j,η,ϵ)
            _J3_ijj = J3_coffee(T,i,j,j,η,ϵ)
            _a_3 += 3*a_3_iij*a_3_ijj*(a_3_i*_J3_iij + a_3_j*_J3_ijj)
            for k ∈ p_comps[idx_i+idx_j+1:end]
                xₖ,μ̄ₖ² = x[k],μ̄²[k]
                _J3_ijk = J3_coffee(T,i,j,k,η,ϵ)
                _a_3 += 6*xᵢ*xⱼ*xₖ*μ̄ᵢ²*μ̄ⱼ²*μ̄ₖ²*σij⁻¹/(σ[i,k]*σ[j,k])*_J3_ijk
            end
        end
    end
    _a_3 *= -4*π^2/3*ρ̄^2/T^3
    return _a_3
end

function J2_coffee(T,i,j,η,ϵ)
    b2 = COFFEEconsts.corr_b2
    c2 = COFFEEconsts.corr_c2
    ϵT = ϵ[i,j]/T
    c = b2 .+ c2 .* ϵT
    return evalpoly(η,c)
end

function J3_coffee(T,i,j,k,η,ϵ)
    b3 = COFFEEconsts.corr_b3
    c3 = COFFEEconsts.corr_c3
    ϵT = ϵ[i,j]/T
    c = b3 .+ c3 .* ϵT
    return evalpoly(η,c)
end

function f_nf(system::DFTSystem, model::COFFEEModel, n, n₃, nᵥ)
    HSd = system.species.size
    (_, T) = system.structure.conditions
    
    ϵ = model.params.epsilon[1]
    σ = diagvalues(model.params.sigma.values)
    d = model.params.shift[1] / σ[1]
    μ² = model.params.dipole2[1]./ϵ/σ[1]^3
    m = model.params.segment
    μ = sqrt(μ²)

    n₀ = zero(first(n) + first(m) + first(HSd))
    n₂, nᵥ₂, η =zero(n₀), zero(n₀), zero(n₀)
    for i in 1:length(n)
        nᵢ,mᵢ,nᵥᵢ,HSdᵢ = n[i],m[i],nᵥ[i],HSd[i]
        nᵢmᵢ = nᵢ*mᵢ
        n₀ += nᵢmᵢ/HSdᵢ
        n₂ += π*HSdᵢ*nᵢmᵢ
        nᵥ₂ += -2π*nᵥᵢ*mᵢ
        η += n₃[i]*mᵢ
    end

    ρ̄ = η*6/π*(σ[1]/HSd[1])^3
    ξ = 1-nᵥ₂^2/n₂^2

    g_hs = 1/(1-η)+HSd[1]*ξ*n₂/(4*(1-η)^2)+HSd[1]^2/4*n₂^2*ξ/(18*(1-η)^3)
    T̄ = T/ϵ

    _Iμμ = Iμμ(ρ̄,T̄,μ)

    Q = ∫∫∫Odξ₁dξ₂dγ12(ρ̄,T̄,μ²,d,_Iμμ)

    return 19*π/12*n₀*ρ̄*g_hs*log(4π/Q)
end

function ∫∫∫Oodξ₁dξ₂dγ12(ρ̄,T̄,μ²,d,_Iμμ,Q)
    I(x) = begin
        _cosΘ = x[1]*x[2]-√(1-x[1]^2)*√(1-x[2]^2)*cos(x[3])
        _I = ∫odr(d,x[1],x[2],x[3])
        return exp(-24/19*μ²/T̄*_Iμμ*_I)*_cosΘ
    end

    _I = ∫∫∫dξ₁dξ₂dγ12(I)

    return _I/Q
end

function cosΘ(system::DFTSystem)
    model = system.model
    HSd = system.species.size
    n = evaluate_field(system)
    η = n[:,2]
    (_, T) = system.structure.conditions
    ϵ = model.params.epsilon[1]
    σ = diagvalues(model.params.sigma.values)
    d = model.params.shift[1] / σ[1]
    μ² = model.params.dipole2[1]./ϵ/σ[1]^3
    μ = sqrt(μ²)
    T̄ = T/ϵ

    ρ̄ = η*6/π*(σ[1]/HSd[1])^3

    _Iμμ = Iμμ.(ρ̄,T̄,μ)

    Q = ∫∫∫Odξ₁dξ₂dγ12.(ρ̄,T̄,μ²,d,_Iμμ)
    return ∫∫∫Oodξ₁dξ₂dγ12.(ρ̄,T̄,μ²,d,_Iμμ,Q)
end