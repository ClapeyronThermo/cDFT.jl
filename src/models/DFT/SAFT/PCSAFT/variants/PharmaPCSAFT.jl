import Clapeyron: pharmaPCSAFTModel, Δσh20, water08_k

function preallocate_params(system::DFTSystem{<:pharmaPCSAFTModel})
    params_base, nc = invoke(preallocate_params, Tuple{DFTSystem{<:PCSAFTModel}}, system)

    model   = system.model
    backend = system.options.device
    T       = system.structure.conditions[2]

    k  = Int(water08_k(model))
    nc_model = length(model)
    sigma_eff = copy(model.params.sigma.values)

    if k > 0
        Δσ = Δσh20(T)
        for i in 1:nc_model, j in 1:nc_model
            sigma_eff[i,j] += (0.5*(k == i) + 0.5*(k == j)) * Δσ
        end
    end

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        eps_vals = model.params.epsilon_assoc.values
        sig3_eff = [sigma_eff[eps_vals.outer_indices[idx]...]^3
                    for idx in 1:length(eps_vals.values)]
        return merge(params_base, (;
            sigma      = Adapt.adapt(backend, sigma_eff),
            assoc_sig3 = Adapt.adapt(backend, sig3_eff),
        )), nc
    else
        return merge(params_base, (;
            sigma = Adapt.adapt(backend, sigma_eff),
        )), nc
    end
end

function Δ(model::pharmaPCSAFTModel, T, n, n₃, nᵥ, i, j, a, b)
    ϵ_assoc = model.params.epsilon_assoc.values
    κ = model.params.bondvol.values
    κijab = κ[i,j][a,b]

    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(κijab))
    iszero(κijab) && return _0
    k = water08_k(model)
    Δσ = Δσh20(T)
    σij = model.params.sigma.values[i,j] + (0.5*(k==i) + 0.5*(k==j))*Δσ
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
    nᵥ₂nᵥ₂ = dot(nᵥ₂,nᵥ₂)

    ξ = 1-nᵥ₂nᵥ₂/n₂^2
    g_hs = 1/(1-n₃₃)+dij*ξ*n₂/(2*(1-n₃₃)^2)+dij^2*n₂^2*ξ/(18*(1-n₃₃)^3)
    return g_hs*σij^3*expm1(ϵ_assoc[i,j][a,b]/T)*κijab
end
