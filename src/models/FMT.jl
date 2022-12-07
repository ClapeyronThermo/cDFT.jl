"""
    F_hs(model::EoSModel, V, T, z=SA[1.0])
Returns the Helmholtz Functional for a Hard-Sphere System

## Description
Hard-Sphere Functional derived using Fundamental Measure Theory as presented by Yu and Wu.
## References
1. Yu, Y-X., & Wu, jl. (2002). Structures of hard-sphere fluids from a modified fundamental-measure theory. The Journal of Chemical Physics, 117(22), 10156-10164. [doi:10.1063/1.1520530](https://doi.org/10.1063/1.1520530)
"""

function F_hs(model::EoSModel,ρ,T,z)
    N_A = Clapeyron.N_A
    d = Clapeyron.d(model,[],T,[1.])[1]
    m = model.params.segment.values[1]

    idx = @. (z[1]+d/2<=z && z<=z[end]-d/2)
    
    dz = z[2]-z[1]
    
    n = ∫fdz.(Ref(ρ),Ref(z),z[idx],d/2)*N_A
    n₃ = ∫fz²dz.(Ref(ρ),Ref(z),z[idx],d/2)*π*m*N_A
    nᵥ = ∫fzdz.(Ref(ρ),Ref(z),z[idx],d/2)*N_A

    n₀ = m/d*n
    n₁ = m/2*n
    n₂ = π*m*d*n
    nᵥ₁ = -m/d*nᵥ
    nᵥ₂ = -2π*m*nᵥ
    
    Φ = @. -n₀*log(1-n₃)+(n₁*n₂-nᵥ₂*nᵥ₁)/(1-n₃)+(n₂^3/3-n₂*nᵥ₂*nᵥ₂)*(log(1-n₃)/(12*π*n₃^2)+1/(12*π*n₃*(1-n₃)^2))

    return ∫(Φ,dz)
end

function δFδρ_hs(model::EoSModel,ρ,T,z)
    N_A = Clapeyron.N_A
    d = Clapeyron.d(model,[],T,[1.])[1]
    m = model.params.segment.values[1]
    
    idx1 = @. (z[1]+d/2<=z && z<=z[end]-d/2)
    
    n = ∫fdz.(Ref(ρ),Ref(z),z[idx1],d/2)*N_A
    n₃ = ∫fz²dz.(Ref(ρ),Ref(z),z[idx1],d/2)*π*m*N_A
    nᵥ = ∫fzdz.(Ref(ρ),Ref(z),z[idx1],d/2)*N_A

    n₀ = m/d*n
    n₁ = m/2*n
    n₂ = π*m*d*n
    nᵥ₁ = -m/d*nᵥ
    nᵥ₂ = -2π*m*nᵥ

    ∂Φ∂n₀ = @. -log(1-n₃)
    ∂Φ∂n₁ = @. n₂/(1-n₃)
    ∂Φ∂n₂ = @. n₁/(1-n₃)+(3*n₂^2-3*nᵥ₂^2)*
               (n₃+(1-n₃)^2*log(1-n₃))/(36π*n₃^2*(1-n₃)^2)
    ∂Φ∂n₃ = @. n₀/(1-n₃)+(n₁*n₂-nᵥ₁*nᵥ₂)/(1-n₃)^2-(n₂^3-3*n₂*nᵥ₁*nᵥ₂)*
               (n₃*(n₃^2-5*n₃+2)+2*(1-n₃)^2*log(1-n₃))/(36π*n₃^3*(1-n₃)^3)
    ∂Φ∂nᵥ₁ = @. -nᵥ₂/(1-n₃)
    ∂Φ∂nᵥ₂ = @. -nᵥ₁/(1-n₃)-n₂*nᵥ₂*(n₃+(1-n₃)^2*log(1-n₃))/(6π*n₃^2*(1-n₃)^2)

    idx2 = @. (z[1]+d<=z && z<=z[end]-d)

    δFδρ_hs_1 = ∫fdz.(Ref(1/d*∂Φ∂n₀+1/2*∂Φ∂n₁+π*d*∂Φ∂n₂),Ref(z[idx1]),z[idx2],d/2)
    δFδρ_hs_2 = ∫fz²dz.(Ref(π*∂Φ∂n₃),Ref(z[idx1]),z[idx2],d/2)
    δFδρ_hs_3 = ∫fzdz.(Ref(1/d*∂Φ∂nᵥ₁+2π*∂Φ∂nᵥ₂),Ref(z[idx1]),z[idx2],d/2)
    return m*(δFδρ_hs_1+δFδρ_hs_2+δFδρ_hs_3)
end

export F_hs, δFδρ_hs
