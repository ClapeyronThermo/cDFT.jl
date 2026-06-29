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
    FV    = F2 + 2     # first vector field index

    n₀=0.0; n₁=0.0; n₂=0.0; n₃₃=0.0
    nv1_1=0.0; nv1_2=0.0; nv1_3=0.0
    nv2_1=0.0; nv2_2=0.0; nv2_3=0.0

    @inbounds for i in 1:NC
        mi = meff[i]; di = HSd[i]
        nim = n[kk, F2, i] * mi
        n₀  += nim / di;  n₁ += 0.5*nim;  n₂ += π*nim*di
        if ND>=1; nvi=n[kk, FV,   i]; nv1_1 += nvi*mi/di; nv2_1 -= 2.0*π*nvi*mi; end
        if ND>=2; nvi=n[kk, FV+1, i]; nv1_2 += nvi*mi/di; nv2_2 -= 2.0*π*nvi*mi; end
        if ND>=3; nvi=n[kk, FV+2, i]; nv1_3 += nvi*mi/di; nv2_3 -= 2.0*π*nvi*mi; end
        n₃₃ += n[kk, F2+1, i] * mi
    end

    nv1v2 = nv1_1*nv2_1 + nv1_2*nv2_2 + nv1_3*nv2_3
    nv2sq = nv2_1*nv2_1 + nv2_2*nv2_2 + nv2_3*nv2_3
    dn     = 1.0 - n₃₃
    log1dn = Base.log(abs(dn))
    inv1dn = 1.0/dn
    bkt    = log1dn/(12.0*π*n₃₃*n₃₃) +
             1.0/(12.0*π*n₃₃*dn*dn)
    res_hs = -n₀*log1dn + (n₁*n₂ - nv1v2)*inv1dn +
             (n₂*n₂*n₂/3.0 - n₂*nv2sq)*bkt
    return res_hs, n₀, n₂, n₃₃, nv2_1, nv2_2, nv2_3
end