export F_hc, δFδρ_hc

function F_hc(model::SAFTFunctionalModel,T)
    m = model.eosmodel.params.segment.values[1]
    z = model.coords
    HSd = d(model.eosmodel,[],T,[1.])[1]
    dz = (z[2]-z[1])*HSd
    
    (n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)  = weights_hs(model,T,z)
    n₂ = n₂/6
    g_hs = @. 1/(1-n₃)+3/2*n₂*HSd/(1-n₃)^2+1/2*n₂^2*HSd^2/(1-n₃)^3
    
    I = @. -n₀*(m-1)*log(g_hs)

    return ∫(I,dz)/∫(n₀,dz)
end

function δFδρ_hc(model::SAFTFunctionalModel,T)
    m = model.eosmodel.params.segment.values[1]
    z = model.coords
    z_full = model.coords_full

    HSd = d(model.eosmodel,[],T,[1.])[1]
    dz = (z[2]-z[1])*HSd
    
    idx1 = @. (z[1]-1<=z_full && z_full<=z[end]+1)

    (n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)  = weights_hs(model,T,z_full[idx1])

    n₂ = n₂/6
    g_hs = @. 1/(1-n₃)+3/2*n₂*HSd/(1-n₃)^2+1/2*n₂^2*HSd^2/(1-n₃)^3
    
    ∂I∂n₀ = @. -(m-1)*log(g_hs)
    ∂I∂n₂ = @. -n₀*(m-1)/g_hs*(3/2*HSd/(1-n₃)^2+n₂*HSd^2/(1-n₃)^3)
    ∂I∂n₃ = @. -n₀*(m-1)/g_hs*(1/(1-n₃)^2+3*n₂*HSd/(1-n₃)^3+3/2*n₂^2*HSd^2/(1-n₃)^4)

    δFδρ_hc_1 = ∫fdz.(Ref(1/HSd*∂I∂n₀+π/6*HSd*∂I∂n₂),Ref(z_full[idx1]),z,1/2)*HSd
    δFδρ_hc_2 = ∫fz²dz.(Ref(π*∂I∂n₃),Ref(z_full[idx1]),z,1/2)*HSd^3
    return δFδρ_hc_1+δFδρ_hc_2
end