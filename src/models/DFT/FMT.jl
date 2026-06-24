"""
    F_hs(model::EoSModel, V, T, z=SA[1.0])
Returns the Helmholtz Functional for a Hard-Sphere System

## Description
Hard-Sphere Functional derived using Fundamental Measure Theory as presented by Yu and Wu.
## References
1. Yu, Y-X., & Wu, J. (2002). Structures of hard-sphere fluids from a modified fundamental-measure theory. The Journal of Chemical Physics, 117(22), 10156-10164. [doi:10.1063/1.1520530](https://doi.org/10.1063/1.1520530)
"""
function f_hs(system::Union{DFTSystem,ElectrolyteDFTSystem}, model::Union{SAFTModel,PeTSModel}, n, n₃, nᵥ)
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

# ── Enzyme / KernelAbstractions kernel: pointwise FMT hard-sphere ─────────
"""
    f_hs(n, meff, HSd, kk, ::Val{NC}, ::Val{ND}, ::Val{F2})

FMT Rosenfeld hard-sphere free energy at grid point `kk`.

- `meff[i]`: effective mass per bead (= m[i] for most models; = m[i]*S[i] for GC models)
- `F2`: field index for the ∫ρdz (0.5d-weighted) scalar density (n₂ source)
  - `F2=2` for PCSAFT, PCPSAFT, SAFTVRMie, SAFTgammaMie, HeterogcPCPSAFT
  - `F2=1` for PeTS, COFFEE
- `F2+1` is the n₃ (volume fraction) field
- `F2+2 .. F2+1+ND` are the vector fields

Returns `(res_hs, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3)`.
Extra moments are needed by COFFEE's near-field polar term.
"""
@inline function f_hs(n, meff, HSd, kk, ::Val{NC}, ::Val{ND}, ::Val{F2}) where {NC, ND, F2}
    _pi   = 3.141592653589793
    eps_v = 1e-15
    FV    = F2 + 2     # first vector field index

    n₀=0.0; n₁=0.0; n₂=0.0; n₃₃=0.0
    nv1_1=0.0; nv1_2=0.0; nv1_3=0.0
    nv2_1=0.0; nv2_2=0.0; nv2_3=0.0

    @inbounds for i in 1:NC
        mi = meff[i]; di = HSd[i]
        nim = n[kk, F2, i] * mi
        n₀  += nim / di;  n₁ += 0.5*nim;  n₂ += _pi*nim*di
        if ND>=1; nvi=n[kk, FV,   i]; nv1_1 += nvi*mi/di; nv2_1 -= 2.0*_pi*nvi*mi; end
        if ND>=2; nvi=n[kk, FV+1, i]; nv1_2 += nvi*mi/di; nv2_2 -= 2.0*_pi*nvi*mi; end
        if ND>=3; nvi=n[kk, FV+2, i]; nv1_3 += nvi*mi/di; nv2_3 -= 2.0*_pi*nvi*mi; end
        n₃₃ += n[kk, F2+1, i] * mi
    end

    nv1v2 = nv1_1*nv2_1 + nv1_2*nv2_2 + nv1_3*nv2_3
    nv2sq = nv2_1*nv2_1 + nv2_2*nv2_2 + nv2_3*nv2_3
    dn     = 1.0 - n₃₃
    log1dn = Base.log(abs(dn) + eps_v)
    inv1dn = 1.0/(dn + eps_v)
    bkt    = log1dn/(12.0*_pi*(n₃₃*n₃₃ + eps_v)) +
             1.0/(12.0*_pi*(n₃₃ + eps_v)*(dn*dn + eps_v))
    res_hs = -n₀*log1dn + (n₁*n₂ - nv1v2)*inv1dn +
             (n₂*n₂*n₂/3.0 - n₂*nv2sq)*bkt
    return res_hs, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3
end