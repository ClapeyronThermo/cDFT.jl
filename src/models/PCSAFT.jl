using Clapeyron: PCSAFTModel

struct PCSAFTSpecies <: DFTSpecies 
    nbeads::Vector{Int64}
    size::Vector{Float64}
end

function get_fields(model::PCSAFTModel)
    nc = length(model)
    return [WeightedDensity(:ρ,zeros(nc)),
            WeightedDensity(:∫ρdz,0.5*ones(nc)),
            WeightedDensity(:∫ρz²dz,0.5*ones(nc)),
            WeightedDensity(:∫ρzdz,0.5*ones(nc)),
            WeightedDensity(:∫ρz²dz,ones(nc)),
            WeightedDensity(:∫ρdz,ones(nc)),
            WeightedDensity(:∫ρz²dz,1.3862*ones(nc))]
end

function get_species(model::PCSAFTModel,structure::DFTStructure)
    (p,T,z) = structure.conditions
    nc = length(model)
    nbeads = ones(nc)
    size = d(model,1e-3,T,z)
    return PCSAFTSpecies(nbeads,size)
end

function f_res(system::DFTSystem, model::PCSAFTModel,n)
    return f_hs(system,model,n[2,:],n[3,:],n[4,:]) + f_hc(system,model,n[1,:],n[5,:],n[6,:]) + f_disp(system,model,n[7,:]) + f_assoc(system,model,n[2,:],n[3,:],n[4,:])
end

function f_hc(system::DFTSystem, model::PCSAFTModel, ρhc, ρ̄hc, _λ)
    HSd = system.species.size
    m = model.params.segment.values
    ζ₃ = zero(eltype(HSd)) + zero(eltype(ρ̄hc))
    ζ₂ = zero(ζ₃)
    for i in @comps
        mi,ρ̄hci,HSdi = m[i],ρ̄hc[i],HSd[i]
        ζ₃ += mi*ρ̄hci
        ζ₂ += mi*ρ̄hci/HSdi
    end
    ζ₃ *= 0.125
    ζ₂ *= 0.125
    #ζ₃ = 1/8*dot(m,ρ̄hc)
    #ζ₂ = sum(1/8*m.*ρ̄hc./HSd)
    ∑f = zero(ζ₃)
    for i in @comps
        λ = _λ[i]/(2*HSd[i])
        yᵈᵈ = 1/(1-ζ₃) + 1.5*HSd[i]*ζ₂/(1-ζ₃)^2+0.5*HSd[i]^2*ζ₂^2/(1-ζ₃)^3
        fi = -ρhc[i]*(m[i]-1)*log(yᵈᵈ*λ/ρhc[i])
        ∑f += fi
    end
    
    return ∑f
end

function f_disp(system::DFTSystem, model::PCSAFTModel, ρ̄)
    HSd = system.species.size
    (_, T, _) = system.structure.conditions
    ψ = 1.3862
    σ = model.params.sigma.values
    m = model.params.segment.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π

    x = ρ̄ /sum(ρ̄)
    m̄ = dot(x,m)

    η = π/6*sum(ρ̄.*m.*HSd.^3)

    C₁ = 1+m̄*(8*η-2*η^2)/(1-η)^4+(1-m̄)*(20*η-27*η^2+12*η^3-2*η^4)/((1-η)^2*(2-η)^2)
    I₁ = I(model,m̄,η,1)
    I₂ = I(model,m̄,η,2)

    m2ϵσ3₁,m2ϵσ3₂ =  Clapeyron.m2ϵσ3(model,zero(T), T, x)
    ρ̄ = sum(ρ̄)
    return -2*π*ρ̄^2*I₁*m2ϵσ3₁-π*ρ̄^2*m̄*C₁^-1*I₂*m2ϵσ3₂
end

function I(model::PCSAFTModel,m̄,n₃,n)
    if n == 1
        corr = Clapeyron.PCSAFTconsts.corr1
    elseif n == 2
        corr = Clapeyron.PCSAFTconsts.corr2
    end
    res = zero(n₃)
    @inbounds for i ∈ 1:7
        ii = i-1 
        corr1,corr2,corr3 = corr[i]
        ki = corr1 + (m̄-1)/m̄*corr2 + (m̄-1)/m̄*(m̄-2)/m̄*corr3
        res += ki*n₃^ii
    end
    return res
end

function f_assoc(system::DFTSystem, model::PCSAFTModel, n, n₃, nᵥ)
    HSd = system.species.size
    (_, T, _) = system.structure.conditions
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ))
    nn = assoc_pair_length(model)
    iszero(nn) && return _0

    n₀ = n./HSd
    n₂ = π.*HSd.*n

    nᵥ₂ = -2π.*nᵥ

    ξ = 1 .-nᵥ₂.^2 ./ n₂.^2

    X_ = X(model,T,n,n₃,nᵥ)
    _0 = zero(first(X_.v))

    ns = model.sites.n_sites
    res = _0
    resᵢₐ = _0
    for i ∈ @comps
        ni = ns[i]
        iszero(length(ni)) && continue
        Xᵢ = X_[i]
        resᵢₐ = _0
        for (a,nᵢₐ) ∈ pairs(ni)
            Xᵢₐ = Xᵢ[a]
            nᵢₐ = ni[a]
            resᵢₐ +=  nᵢₐ* (log(Xᵢₐ) - Xᵢₐ/2 + 0.5)
        end
        res += resᵢₐ*n₀[i]*ξ[i]
    end
    return res
end

function Δ(model::PCSAFTModel, T, n, n₃, nᵥ, i, j, a, b)
    ϵ_assoc = model.params.epsilon_assoc.values
    κ = model.params.bondvol.values
    κijab = κ[i,j][a,b] 
    iszero(κijab) && return _0

    σ = model.params.sigma.values[i,j]
    m = model.params.segment.values
    HSd = d(model,nothing,T,onevec(model))
    dij = (HSd[i]*HSd[j])/(HSd[i]+HSd[j])

    n₂ = sum(π.*HSd.*n.*m)
    nᵥ₂ = sum(-2π.*nᵥ.*m)
    n₃  = sum(n₃.*m)

    ξ = 1-nᵥ₂^2/n₂^2
    g_hs = 1/(1-n₃)+dij*ξ*n₂/(2*(1-n₃)^2)+dij^2*n₂^2*ξ/(18*(1-n₃)^3)
    return g_hs*σ^3*(exp(ϵ_assoc[i,j][a,b]/T)-1)*κijab
end

export F_res, δFδρ_res

function length_scale(model::SAFTModel)
    return maximum(model.params.sigma.values)
end