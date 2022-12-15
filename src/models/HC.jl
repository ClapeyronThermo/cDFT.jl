export F_hc, δFδρ_hc

function F_hc(model::SAFTFunctionalModel,T)
    m = model.eosmodel.params.segment.values[1]
    z = model.coords
    HSd = d(model.eosmodel,[],T,[1.])[1]
    dz = (z[2]-z[1])*HSd
    
    (n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)  = weights_hs(model,T,z)
    
    I = f_hc(Ref(model), T, n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)

    return ∫(I,dz)/∫(n₀,dz)
end

function f_hc(model::SAFTFunctionalModel, T, n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)
    m = model.eosmodel.params.segment.values[1]
    HSd = d(model.eosmodel,[],T,[1.])[1]
    
    N₂ = n₂/6
    g_hs = @. 1/(1-n₃)+3/2*N₂*HSd/(1-n₃)^2+1/2*N₂^2*HSd^2/(1-n₃)^3
    
    return -n₀*(m-1)*log(g_hs)
end

function δfδρ_hc(model::SAFTFunctionalModel ,T ,n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)    
    f(x) = f_hc(model,T,x[1],x[2],x[3],x[4],x[5],x[6])
    df(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(df,hcat([n₀ n₁ n₂ n₃ nᵥ₁ nᵥ₂]);dims=2)
    ∂f∂n₀ = δfδn[:,1]
    ∂f∂n₁ = δfδn[:,2]
    ∂f∂n₂ = δfδn[:,3]
    ∂f∂n₃ = δfδn[:,4]
    ∂f∂nᵥ₁ = δfδn[:,5]
    ∂f∂nᵥ₂ = δfδn[:,6]
    
    return (∂f∂n₀, ∂f∂n₁, ∂f∂n₂, ∂f∂n₃, ∂f∂nᵥ₁, ∂f∂nᵥ₂)
end


function δFδρ_hc(model::SAFTFunctionalModel,T)
    HSd = d(model.eosmodel,[],T,[1.])[1]

    z = model.coords
    z_full = model.coords_full
    
    idx1 = @. (z[1]-1<=z_full && z_full<=z[end]+1)

    (n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)  = weights_hs(model,T,z_full[idx1])
    (∂f∂n₀, ∂f∂n₁, ∂f∂n₂, ∂f∂n₃, ∂f∂nᵥ₁, ∂f∂nᵥ₂) = δfδρ_hc(model, T, n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)

    δFδρ_hc_1 = ∫fdz.(Ref(1/HSd*∂f∂n₀+1/2*∂f∂n₁+π*HSd*∂f∂n₂),Ref(z_full[idx1]),z,1/2)*HSd
    δFδρ_hc_2 = ∫fz²dz.(Ref(π*∂f∂n₃),Ref(z_full[idx1]),z,1/2)*HSd^3
    δFδρ_hc_3 = ∫fzdz.(Ref(1/HSd*∂f∂nᵥ₁+2π*∂f∂nᵥ₂),Ref(z_full[idx1]),z,1/2)*HSd^2

    return δFδρ_hc_1+δFδρ_hc_2+δFδρ_hc_3
end

# function δFδρ_hc(model::SAFTFunctionalModel,T)
#     m = model.eosmodel.params.segment.values[1]
#     z = model.coords
#     z_full = model.coords_full

#     HSd = d(model.eosmodel,[],T,[1.])[1]
#     dz = (z[2]-z[1])*HSd
    
#     idx1 = @. (z[1]-1<=z_full && z_full<=z[end]+1)

#     (n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)  = weights_hs(model,T,z_full[idx1])

#     n₂ = n₂/6
#     g_hs = @. 1/(1-n₃)+3/2*n₂*HSd/(1-n₃)^2+1/2*n₂^2*HSd^2/(1-n₃)^3
    
#     ∂I∂n₀ = @. -(m-1)*log(g_hs)
#     ∂I∂n₂ = @. -n₀*(m-1)/g_hs*(3/2*HSd/(1-n₃)^2+n₂*HSd^2/(1-n₃)^3)
#     ∂I∂n₃ = @. -n₀*(m-1)/g_hs*(1/(1-n₃)^2+3*n₂*HSd/(1-n₃)^3+3/2*n₂^2*HSd^2/(1-n₃)^4)

#     δFδρ_hc_1 = ∫fdz.(Ref(1/HSd*∂I∂n₀+π/6*HSd*∂I∂n₂),Ref(z_full[idx1]),z,1/2)*HSd
#     δFδρ_hc_2 = ∫fz²dz.(Ref(π*∂I∂n₃),Ref(z_full[idx1]),z,1/2)*HSd^3
#     return δFδρ_hc_1+δFδρ_hc_2
# end