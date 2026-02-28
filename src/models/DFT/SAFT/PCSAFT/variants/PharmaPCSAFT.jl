import Clapeyron: pharmaPCSAFTModel, Δσh20, water08_k

function Δ(model::pharmaPCSAFTModel, T, n, n₃, nᵥ, i, j, a, b)
    ϵ_assoc = model.params.epsilon_assoc.values
    κ = model.params.bondvol.values
    κijab = κ[i,j][a,b]

    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(κijab))
    iszero(κijab) && return _0
    k = water08_k(model)
    Δσ = Δσh20(T)
    σij = model.params.sigma.values[i,j][i,j] + (0.5*(k==i) + 0.5*(k==j))*Δσ
    m = model.params.segment.values
    HSd = d(model,1e-3,T,onevec(model))
    dij = (HSd[i]*HSd[j])/(HSd[i]+HSd[j])

    n₂, nᵥ₂, n₃₃ = _0,zero(nᵥ[:,1]),_0
    for i in 1:length(n)
        nᵢ,mᵢ,nᵥᵢ,HSdᵢ = n[i],m[i],nᵥ[:,i],HSd[i]
        n₂ += π*HSdᵢ*nᵢ*mᵢ
        nᵥ₂ .+= -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[i]*mᵢ
    end
    #n₂ = sum(π.*HSd.*n.*m)
    nᵥ₂nᵥ₂ = dot(nᵥ₂,nᵥ₂)
    #n₃  = sum(n₃.*m)

    ξ = 1-nᵥ₂nᵥ₂/n₂^2
    g_hs = 1/(1-n₃₃)+dij*ξ*n₂/(2*(1-n₃₃)^2)+dij^2*n₂^2*ξ/(18*(1-n₃₃)^3)
    return g_hs*σij^3*expm1(ϵ_assoc[i,j][a,b]/T)*κijab
end