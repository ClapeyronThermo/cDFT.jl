"""
    F_hs(model::EoSModel, V, T, z=SA[1.0])
Returns the Helmholtz Functional for a Hard-Sphere System

## Description
Hard-Sphere Functional derived using Fundamental Measure Theory as presented by Yu and Wu.
## References
1. Yu, Y-X., & Wu, J. (2002). Structures of hard-sphere fluids from a modified fundamental-measure theory. The Journal of Chemical Physics, 117(22), 10156-10164. [doi:10.1063/1.1520530](https://doi.org/10.1063/1.1520530)
"""

function F_hs(model::SAFTModel,ρ,T,z)
    HSd = d(model,[],T,[1.])[1]
    dz = ρ.mesh_size

    lim = 1/2*HSd

    (n, n₃,nᵥ)  = weights_hs(model,ρ,z,lim)

    n₀ = n./HSd
    
    Φ = f_hs.(Ref(model), Ref(T), n, n₃, nᵥ)

    return ∫(Φ,dz) ./ ∫(n₀,dz)
end

function f_hs(model::SAFTModel, T, n, n₃, nᵥ)
    HSd = d(model,[],T,[1.])[1]

    n₀ = n./HSd
    n₁ = n./2
    n₂ = π.*HSd.*n

    nᵥ₁ = -nᵥ./HSd
    nᵥ₂ = -2π.*nᵥ

    m = model.params.segment.values[1]
    return m*(-n₀*log(1-n₃)+(n₁*n₂-nᵥ₂*nᵥ₁)/(1-n₃)+(n₂^3/3-n₂*nᵥ₂*nᵥ₂)*(log(1-n₃)/(12*π*n₃^2)+1/(12*π*n₃*(1-n₃)^2)))
end

function δfδρ_hs(model::SAFTModel ,T ,n, n₃, nᵥ)    
    f(x) = f_hs(model,T,x[1],x[2],x[3])
    df(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(df,hcat([n n₃ nᵥ]);dims=2)
    ∂f∂n = δfδn[:,1]
    ∂f∂n₃ = δfδn[:,2]
    ∂f∂nᵥ = δfδn[:,3]
    
    return (∂f∂n, ∂f∂n₃, ∂f∂nᵥ)
end

function δFδρ_hs(model::SAFTModel,ρ,T,z)
    HSd = d(model,[],T,[1.])[1]
    lim = 1/2*HSd
    bounds = ρ.bounds.+[-lim,lim]
    mesh_size = ρ.mesh_size

    (n, n₃, nᵥ)  = weights_hs(model,ρ,z,lim)

    z_damp = 0:mesh_size:bounds[2]
    zu = [z_damp[i] for i in 1:length(z_damp)]
    zd = [-z_damp[i] for i in length(z_damp):-1:2]
    z_damp = vcat(zd,zu)

    (∂f∂n, ∂f∂n₃, ∂f∂nᵥ) = δfδρ_hs(model, T, n, n₃, nᵥ)

    ∂f∂n = DensityProfile(∂f∂n,z,bounds,[∂f∂n[1],∂f∂n[end]])
    ∂f∂n₃ = DensityProfile(∂f∂n₃,z,bounds,[∂f∂n₃[1],∂f∂n₃[end]])
    ∂f∂nᵥ = DensityProfile(∂f∂nᵥ,z,bounds,[∂f∂nᵥ[1],∂f∂nᵥ[end]])

    span = range(-lim,lim,length=101)

    δFδρ_hs_1 = ∫ρdz.(Ref(∂f∂n),z,Ref(span))
    δFδρ_hs_2 = π*∫ρz²dz.(Ref(∂f∂n₃),z,Ref(span))
    δFδρ_hs_3 = -∫ρzdz.(Ref(∂f∂nᵥ),z,Ref(span))

    return δFδρ_hs_1+δFδρ_hs_2+δFδρ_hs_3
end

# function δFδρ_hs(model::SAFTFunctionalModel,T)
#     z = model.coords
#     z_full = model.coords_full

#     HSd = d(model.eosmodel,[],T,[1.])[1]
#     dz = (z[2]-z[1])*HSd

#     m = model.eosmodel.params.segment.values[1]
    
#     idx1 = @. (z[1]-1<=z_full && z_full<=z[end]+1)

#     (n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)  = weights_hs(model,T,z_full[idx1])

#     ∂Φ∂n₀ = @. -log(1-n₃)
#     ∂Φ∂n₁ = @. n₂/(1-n₃)
#     ∂Φ∂n₂ = @. n₁/(1-n₃)+3*(n₂^2-nᵥ₂^2)*
#                (n₃+(1-n₃)^2*log(1-n₃))/(36π*n₃^2*(1-n₃)^2)
#     ∂Φ∂n₃ = @. n₀/(1-n₃)+(n₁*n₂-nᵥ₁*nᵥ₂)/(1-n₃)^2-(n₂^3-3*n₂*nᵥ₂^2)*
#                (n₃*(n₃^2-5*n₃+2)+2*(1-n₃)^3*log(1-n₃))/(36π*n₃^3*(1-n₃)^3)
#     ∂Φ∂nᵥ₁ = @. -nᵥ₂/(1-n₃)
#     ∂Φ∂nᵥ₂ = @. -nᵥ₁/(1-n₃)-n₂*nᵥ₂*(n₃+(1-n₃)^2*log(1-n₃))/(6π*n₃^2*(1-n₃)^2)

#     δFδρ_hs_1 = ∫fdz.(Ref(1/HSd*∂Φ∂n₀+1/2*∂Φ∂n₁+π*HSd*∂Φ∂n₂),Ref(z_full[idx1]),z,1/2)*HSd
#     δFδρ_hs_2 = ∫fz²dz.(Ref(π*∂Φ∂n₃),Ref(z_full[idx1]),z,1/2)*HSd^3
#     δFδρ_hs_3 = ∫fzdz.(Ref(1/HSd*∂Φ∂nᵥ₁+2π*∂Φ∂nᵥ₂),Ref(z_full[idx1]),z,1/2)*HSd^2
#     return m*(δFδρ_hs_1+δFδρ_hs_2+δFδρ_hs_3)
# end

# function weights_hs_old(model::SAFTFunctionalModel,T,z_eval)
#     z_full = model.coords_full

#     HSd = d(model.eosmodel,[],T,[1.])[1]

#     ρ = model.density
#     ρ_full = model.density_full

#     m = model.eosmodel.params.segment.values[1]

#     n = ∫fdz.(Ref(ρ_full),Ref(z_full),z_eval,1/2)*N_A*HSd
#     n₃ = ∫fz²dz.(Ref(ρ_full),Ref(z_full),z_eval,1/2)*π*m*N_A*HSd^3
#     nᵥ = ∫fzdz.(Ref(ρ_full),Ref(z_full),z_eval,1/2)*N_A*HSd^2

#     n₀ = m/HSd*n
#     n₁ = m/2*n
#     n₂ = π*m*HSd*n
#     nᵥ₁ = -m/HSd*nᵥ
#     nᵥ₂ = -2π*m*nᵥ
#     return n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂
# end

export F_hs, δFδρ_hs
