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
    HSd = species.size

    n₀ = zero(first(n) + first(m) + first(HSd))
    n₁,n₂,nᵥ₁,nᵥ₂,n₃₃ = zero(n₀), zero(n₀), zero(nᵥ[:,1]), zero(nᵥ[:,1]), zero(n₀)
    for i in 1:length(n)
        mᵢ,HSdᵢ,nᵥᵢ = m[i],HSd[i],nᵥ[:,i]
        nᵢmᵢ = n[i]*mᵢ
        n₀ += nᵢmᵢ/HSdᵢ
        n₁ += 0.5nᵢmᵢ
        n₂ += π*nᵢmᵢ*HSdᵢ
        nᵥ₁ .+= nᵥᵢ*mᵢ/HSdᵢ
        nᵥ₂ .+= -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[i]*mᵢ
    end
    nᵥ₁nᵥ₂ = dot(nᵥ₁,nᵥ₂)
    nᵥ₂nᵥ₂ = dot(nᵥ₂,nᵥ₂)
    return -n₀*log(1-n₃₃)+(n₁*n₂-nᵥ₁nᵥ₂)/(1-n₃₃)+(n₂^3/3-n₂*nᵥ₂nᵥ₂)*(log(1-n₃₃)/(12*π*n₃₃^2)+1/(12*π*n₃₃*(1-n₃₃)^2))
end