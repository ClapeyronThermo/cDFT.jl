export F_hc, δFδρ_hc

function F_hc(model::SAFTModel,ρ,T,z)
    HSd = d(model,[],T,[1.])[1]
    dz = ρ.mesh_size

    lim = 1/2*HSd

    (n, n₃,nᵥ)  = weights_hs(model,ρ,z,lim)

    n₀ = n./HSd
    
    I = f_hc.(Ref(model), T, n, n₃, nᵥ)

    return ∫(I,dz)/∫(n₀,dz)
end

function f_hc(model::SAFTModel, T, n, n₃, nᵥ)
    m = model.params.segment.values[1]
    HSd = d(model,[],T,[1.])[1]
    
    n₀ = n./HSd
    n₂ = π.*HSd.*n/6
    g_hs = @. 1/(1-n₃)+3/2*n₂*HSd/(1-n₃)^2+1/2*n₂^2*HSd^2/(1-n₃)^3
    
    return -n₀*(m-1)*log(g_hs)
end

function δfδρ_hc(model::SAFTModel , T, n, n₃, nᵥ)
    f(x) = f_hc(model,T,x[1],x[2],x[3])
    df(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(df,hcat([n n₃ nᵥ]);dims=2)
    ∂f∂n = δfδn[:,1]
    ∂f∂n₃ = δfδn[:,2]
    ∂f∂nᵥ = δfδn[:,3]
    
    return (∂f∂n, ∂f∂n₃, ∂f∂nᵥ)
end


function δFδρ_hc(model::SAFTModel,ρ,T,z)
    HSd = d(model,[],T,[1.])[1]
    lim = 1/2*HSd
    bounds = ρ.bounds.+[-lim,lim]
    mesh_size = ρ.mesh_size

    (n, n₃, nᵥ)  = weights_hs(model,ρ,z,lim)

    z_damp = 0:mesh_size:bounds[2]
    zu = [z_damp[i] for i in 1:length(z_damp)]
    zd = [-z_damp[i] for i in length(z_damp):-1:2]
    z_damp = vcat(zd,zu)

    (∂f∂n, ∂f∂n₃, ∂f∂nᵥ) = δfδρ_hc(model, T, n, n₃, nᵥ)

    ∂f∂n = DensityProfile(∂f∂n,z_damp,bounds,[∂f∂n[1],∂f∂n[end]])
    ∂f∂n₃ = DensityProfile(∂f∂n₃,z_damp,bounds,[∂f∂n₃[1],∂f∂n₃[end]])
    ∂f∂nᵥ = DensityProfile(∂f∂nᵥ,z_damp,bounds,[∂f∂nᵥ[1],∂f∂nᵥ[end]])

    span = range(-lim,lim,length=101)

    δFδρ_hc_1 = ∫ρdz.(Ref(∂f∂n),z,Ref(span))
    δFδρ_hc_2 = π*∫ρz²dz.(Ref(∂f∂n₃),z,Ref(span))
    δFδρ_hc_3 = -∫ρzdz.(Ref(∂f∂nᵥ),z,Ref(span))
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