"""
    F_hs(model::EoSModel, V, T, z=SA[1.0])
Returns the Helmholtz Functional for a Hard-Sphere System

## Description
Hard-Sphere Functional derived using Fundamental Measure Theory as presented by Yu and Wu.
## References
1. Yu, Y-X., & Wu, J. (2002). Structures of hard-sphere fluids from a modified fundamental-measure theory. The Journal of Chemical Physics, 117(22), 10156-10164. [doi:10.1063/1.1520530](https://doi.org/10.1063/1.1520530)
"""
function f_hs(system::DFTSystem, model::SAFTModel, n, n₃, nᵥ)
    species = system.species
    m = model.params.segment.values
    HSd = [species[i].size[1] for i in @comps]

    n₀ = zero(first(n) + first(m) + first(HSd))
    n₁,n₂,nᵥ₁,nᵥ₂,n₃₃ = zero(n₀), zero(n₀), zero(n₀), zero(n₀), zero(n₀)
    for i in 1:length(n)
        mᵢ,HSdᵢ,nᵥᵢ = m[i],HSd[i],nᵥ[i]
        nᵢmᵢ = n[i]*mᵢ
        n₀ += nᵢmᵢ/HSdᵢ
        n₁ += 0.5nᵢmᵢ
        n₂ += π*nᵢmᵢ*HSdᵢ
        nᵥ₁ += nᵥᵢ*mᵢ/HSdᵢ
        nᵥ₂ += -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[i]*mᵢ
    end
    return -n₀*log(1-n₃₃)+(n₁*n₂-nᵥ₂*nᵥ₁)/(1-n₃₃)+(n₂^3/3-n₂*nᵥ₂*nᵥ₂)*(log(1-n₃₃)/(12*π*n₃₃^2)+1/(12*π*n₃₃*(1-n₃₃)^2))
end

function δfδρ_hs(model::SAFTModel ,T ,n, n₃, nᵥ)    
    nc = length(model)
    idx = 1:nc
    f(x) = f_hs(model,T,x[idx],x[idx.+nc],x[idx.+2*nc])
    df(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(df,hcat([n n₃ nᵥ]);dims=2)
    ∂f∂n = @view δfδn[:,idx]
    ∂f∂n₃ = @view δfδn[:,idx.+nc]
    ∂f∂nᵥ = @view δfδn[:,idx.+2*nc]
    
    return (∂f∂n, ∂f∂n₃, ∂f∂nᵥ)
end

function δFδρ_hs(model::SAFTModel,ρ,T,z)
    HSd = d(model,1e-3,T,ones(length(model)))
    lim = 1/2*HSd

    (n, n₃, nᵥ)  = weights_hs(model,ρ,z,lim)

    (∂f∂n0, ∂f∂n₃0, ∂f∂nᵥ0) = δfδρ_hs(model, T, n, n₃, nᵥ)
    δFδρ_hs = zeros(length(z),length(model))
    for i in @comps 
        bounds = ρ[i].bounds.+[-lim[i],lim[i]]
        ∂f∂n = DensityProfile(∂f∂n0[:,i],z,bounds,[∂f∂n0[1,i],∂f∂n0[end,i]])
        ∂f∂n₃ = DensityProfile(∂f∂n₃0[:,i],z,bounds,[∂f∂n₃0[1,i],∂f∂n₃0[end,i]])
        ∂f∂nᵥ = DensityProfile(∂f∂nᵥ0[:,i],z,bounds,[∂f∂nᵥ0[1,i],∂f∂nᵥ0[end,i]])
    
        span = range(-lim[i],lim[i],length=101)

        δFδρ_hs_1 = ∫ρdz.(Ref(∂f∂n),z,Ref(span))
        δFδρ_hs_2 = π*∫ρz²dz.(Ref(∂f∂n₃),z,Ref(span))
        δFδρ_hs_3 = -∫ρzdz.(Ref(∂f∂nᵥ),z,Ref(span))

        δFδρ_hs[:,i] = δFδρ_hs_1+δFδρ_hs_2+δFδρ_hs_3
    end
    return δFδρ_hs
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
