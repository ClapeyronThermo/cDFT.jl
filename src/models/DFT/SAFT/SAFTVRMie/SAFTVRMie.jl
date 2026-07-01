using Clapeyron: SAFTVRMieModel
using Clapeyron: aS_1, B, KHS, Cλ, f123456
using Clapeyron: KHS_fdf, aS_1_fdf, B_fdf, g_HS
using Clapeyron: SAFTVRMieconsts

struct SAFTVRMieSpecies <: DFTSpecies 
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_fields(model::SAFTVRMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    nc = length(model)
    ngrid = structure.ngrid
    ω = structure_ω(structure, device, FP)
    d = species.size

    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = species.size ./ σ
    ψ = @. cbrt(3*C*x^3*(x^-λ_a/(λ_a-3)-x^-λ_r/(λ_r-3)))

    return [SWeightedDensity(:ρ,zeros(nc),ω,ngrid,device),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,device),
            SWeightedDensity(:∫ρdz,d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,device)]
end

function get_species(model::SAFTVRMieModel,structure::DFTStructure)
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    size = d(model,1e-3,temperature,ρbulk)

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature
    nc = length(model)
    return SAFTVRMieSpecies(ones(Int64,nc),size,ρbulk,μres)
end

function get_propagator(model::SAFTVRMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return IdealPropagator()
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────
# GPU-safe constants: row-tuple form of the SAFTγMie A matrix for ζeff.
const SAFTVRMIE_A = (
    ( 0.81096,    1.7888,   -37.578,   92.284),
    ( 1.02050,  -19.341,   151.26,   -463.50),
    (-1.90570,   22.845,  -228.14,   973.92),
    ( 1.08850,   -6.1962,  106.98,  -677.64),
)

"""
Pointwise residual free energy for SAFT-VR Mie: FMT hard-sphere + chain + dispersion + association.

Field layout (same as PCSAFTModel):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc  (chain)
  5+ND     : ∫ρdz  with d    → λ    (chain)
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z   (dispersion)
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: SAFTVRMieModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_chain = f_chain(M, kk, n, params, T, Val(NC), Val(ND))
    res_disp  = f_disp(M, kk, n, params, T, Val(NC), Val(ND), Val(6+ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_chain + res_disp + res_assoc
    return nothing
end

"""
SAFT-VR Mie chain contribution (gMie contact value) at grid point `kk`.
"""
@inline function f_chain(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: SAFTVRMieModel}
    HSd     = params.HSd
    m_seg   = params.m
    σ       = params.sigma
    ϵ       = params.epsilon
    λr_mat  = params.lambda_r_t
    λa_mat  = params.lambda_a_t
    A       = params.A
    ϕ       = params.phi

    idx_ζ = 4+ND;  idx_λ = 5+ND

    ρS_c = 0.0
    @inbounds for i in 1:NC
        ρS_c += n[kk,idx_ζ,i] * 3.0/(4.0*π*HSd[i]^3) * m_seg[i]
    end
    kρS_c = ρS_c * π/6.0/8.0

    ζ_Xc = 0.0;  σ3_xc = 0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄hci = n[kk,idx_ζ,i] * 3.0/(4.0*π*di^3)
        x_Si = ρ̄hci * m_seg[i] / ρS_c
        @inbounds for j in 1:NC
            dj   = HSd[j]
            ρ̄hcj = n[kk,idx_ζ,j] * 3.0/(4.0*π*dj^3)
            x_Sj = ρ̄hcj * m_seg[j] / ρS_c
            σ3_xc += x_Si*x_Sj*σ[i,j]^3
            ζ_Xc  += kρS_c*x_Si*x_Sj*(di+dj)^3
        end
    end
    ζstc = σ3_xc * ρS_c * π/6.0

    _KHSc, _∂KHSc = _KHS_fdf_kernel(ρS_c, ζ_Xc)

    res_chain = 0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄hci = n[kk,idx_ζ,i] * 3.0/(4.0*π*di^3)
        x_Si = ρ̄hci * m_seg[i] / ρS_c

        λa = λa_mat[i][i];  λr = λr_mat[i][i]
        _C = _Cλ_kernel(λa, λr)
        x0 = σ[i,i] / di
        ϵii = ϵ[i,i]

        aS1_a,  dS1_a  = _aS1_fdf_kernel(λa,     ζ_Xc, A)
        aS1_r,  dS1_r  = _aS1_fdf_kernel(λr,     ζ_Xc, A)
        B_a,    dB_a   = _B_fdf_kernel(λa,    x0, ζ_Xc)
        B_r,    dB_r   = _B_fdf_kernel(λr,    x0, ζ_Xc)
        aS1_2a, dS1_2a = _aS1_fdf_kernel(2.0*λa,  ζ_Xc, A)
        aS1_2r, dS1_2r = _aS1_fdf_kernel(2.0*λr,  ζ_Xc, A)
        aS1_ar, dS1_ar = _aS1_fdf_kernel(λa+λr,   ζ_Xc, A)
        B_2a,   dB_2a  = _B_fdf_kernel(2.0*λa, x0, ζ_Xc)
        B_2r,   dB_2r  = _B_fdf_kernel(2.0*λr, x0, ζ_Xc)
        B_ar,   dB_ar  = _B_fdf_kernel(λa+λr,  x0, ζ_Xc)

        ∂a1ρS = _C*(x0^λa*(dS1_a+dB_a) - x0^λr*(dS1_r+dB_r))
        g1_   = 3.0*∂a1ρS - _C*(λa*x0^λa*(aS1_a+B_a) - λr*x0^λr*(aS1_r+B_r))

        α   = _C*(1.0/(λa-3.0) - 1.0/(λr-3.0))
        f1,f2,f3,f4,f5,f6 = _f123456_kernel(α, ϕ)
        θ   = exp(ϵii/T) - 1.0
        γc  = 10.0*(-tanh(10.0*(0.57-α))+1.0)*ζstc*θ*exp(-6.7*ζstc-8.0*ζstc^2)

        cb2a = x0^(2.0*λa)*(aS1_2a+B_2a)
        cbar = x0^(λa+λr)*(aS1_ar+B_ar)
        cb2r = x0^(2.0*λr)*(aS1_2r+B_2r)
        ∂a2ρS = 0.5*_C*_C*(
            ρS_c*_∂KHSc*(cb2a - 2.0*cbar + cb2r)
          + _KHSc*(x0^(2.0*λa)*(dS1_2a+dB_2a)
                 - 2.0*x0^(λa+λr)*(dS1_ar+dB_ar)
                 + x0^(2.0*λr)*(dS1_2r+dB_2r))
        )
        gMCA2 = 3.0*∂a2ρS - _KHSc*_C*_C*(
                  λr*cb2r - (λa+λr)*cbar + λa*cb2a)
        g2_ = (1.0+γc)*gMCA2

        gHS   = _gHS_kernel(x0, ζ_Xc)
        gMie  = gHS * exp(ϵii/T * g1_/gHS + (ϵii/T)^2 * g2_/gHS)

        ρhci  = n[kk,1,i]
        λfld  = n[kk,idx_λ,i] / (2.0*di)
        res_chain += ρhci * Base.log(abs(gMie*λfld/ρhci)) * (m_seg[i]-1.0)
    end
    return -res_chain
end

@inline function _Cλ_kernel(λa, λr)
    return (λr / (λr - λa)) * (λr / λa)^(λa / (λr - λa))
end

# ζeff(λ, ζ_X) using SAFTVRMIE_A (row-tuples)
@inline function _ζeff_kernel(λ, ζ_X, A)
    li = 1.0 / λ;  li2 = li*li;  li3 = li2*li
    aλ1 = A[1][1] + A[1][2]*li + A[1][3]*li2 + A[1][4]*li3
    aλ2 = A[2][1] + A[2][2]*li + A[2][3]*li2 + A[2][4]*li3
    aλ3 = A[3][1] + A[3][2]*li + A[3][3]*li2 + A[3][4]*li3
    aλ4 = A[4][1] + A[4][2]*li + A[4][3]*li2 + A[4][4]*li3
    ζ2 = ζ_X*ζ_X;  ζ3 = ζ2*ζ_X;  ζ4 = ζ3*ζ_X
    return aλ1*ζ_X + aλ2*ζ2 + aλ3*ζ3 + aλ4*ζ4
end

@inline function _aS1_kernel(λ, ζ_X, A)
    ζeff = _ζeff_kernel(λ, ζ_X, A)
    return -1.0/(λ-3.0) * (1.0 - ζeff*0.5) / (1.0-ζeff)^3
end

# Returns (aS_1, d(ρS·aS_1)/dρS)
@inline function _aS1_fdf_kernel(λ, ζ_X, A)
    li = 1.0/λ;  li2 = li*li;  li3 = li2*li
    aλ1 = A[1][1] + A[1][2]*li + A[1][3]*li2 + A[1][4]*li3
    aλ2 = A[2][1] + A[2][2]*li + A[2][3]*li2 + A[2][4]*li3
    aλ3 = A[3][1] + A[3][2]*li + A[3][3]*li2 + A[3][4]*li3
    aλ4 = A[4][1] + A[4][2]*li + A[4][3]*li2 + A[4][4]*li3
    ζ2 = ζ_X*ζ_X;  ζ3 = ζ2*ζ_X;  ζ4 = ζ3*ζ_X
    ζeff = aλ1*ζ_X + aλ2*ζ2 + aλ3*ζ3 + aλ4*ζ4
    # ∂ζeff/∂ζ_X * ζ_X  (= ρS·∂ζeff/∂ρS since ζ_X ∝ ρS)
    dζeff_ζ = aλ1*ζ_X + 2.0*aλ2*ζ2 + 3.0*aλ3*ζ3 + 4.0*aλ4*ζ4
    ζeff3   = (1.0-ζeff)^3
    ζeffm1  = 1.0 - ζeff*0.5
    ζf      = ζeffm1 / ζeff3
    λf      = -1.0 / (λ-3.0)
    f       = λf * ζf
    dζf     = (3.0*ζeffm1*(1.0-ζeff)^2 - 0.5*ζeff3) / (ζeff3*ζeff3)
    df      = λf * (ζf + dζeff_ζ * dζf)
    return f, df
end

@inline function _B_kernel(λ, x_0, ζ_X)
    x3λ = x_0^(3.0-λ)
    ζX3 = (1.0-ζ_X)^3
    I   = (1.0 - x3λ) / (λ-3.0)
    J   = (1.0 - (λ-3.0)*x_0^(4.0-λ) + (λ-4.0)*x3λ) / ((λ-3.0)*(λ-4.0))
    return I*(1.0-ζ_X*0.5)/ζX3 - 9.0*J*ζ_X*(ζ_X+1.0)/(2.0*ζX3)
end

# Returns (B, d(ρS·B)/dρS)
@inline function _B_fdf_kernel(λ, x_0, ζ_X)
    x3λ = x_0^(3.0-λ)
    ζX2 = (1.0-ζ_X)^2;  ζX3 = ζX2*(1.0-ζ_X);  ζX6 = ζX3*ζX3
    I   = (1.0 - x3λ) / (λ-3.0)
    J   = (1.0 - (λ-3.0)*x_0^(4.0-λ) + (λ-4.0)*x3λ) / ((λ-3.0)*(λ-4.0))
    f   = I*(1.0-ζ_X*0.5)/ζX3 - 9.0*J*ζ_X*(ζ_X+1.0)/(2.0*ζX3)
    df  = f + ζ_X*(
              (3.0*(1.0-ζ_X*0.5)*ζX2 - 0.5*ζX3)*I/ζX6
            - 9.0*J*((1.0+2.0*ζ_X)*ζX3 + ζ_X*(1.0+ζ_X)*3.0*ζX2)/(2.0*ζX6)
          )
    return f, df
end

@inline function _KHS_kernel(ζ_X)
    return (1.0-ζ_X)^4 / evalpoly(ζ_X, (1.0, 4.0, 4.0, -4.0, 1.0))
end

# Returns (KHS, dKHS/dρS)
@inline function _KHS_fdf_kernel(ρS, ζ_X)
    ζX4   = (1.0-ζ_X)^4
    denom = evalpoly(ζ_X, (1.0, 4.0, 4.0, -4.0, 1.0))
    ddenom = evalpoly(ζ_X, (4.0, 8.0, -12.0, 4.0))
    f    = ζX4 / denom
    ρdf  = -ζ_X*(4.0*(1.0-ζ_X)^3*denom + ζX4*ddenom) / (denom*denom)
    return f, ρdf / ρS
end

@inline function _gHS_kernel(x_0, ζ_X)
    ζX3 = (1.0-ζ_X)^3
    k0 = -Base.log(1.0-ζ_X) + evalpoly(ζ_X,(0.0,42.0,-39.0,9.0,-2.0))/(6.0*ζX3)
    k1 = evalpoly(ζ_X,(0.0,-12.0,6.0,0.0,1.0)) / (2.0*ζX3)
    k2 = -3.0*ζ_X*ζ_X / (8.0*(1.0-ζ_X)^2)
    k3 = evalpoly(ζ_X,(0.0,3.0,3.0,0.0,-1.0)) / (6.0*ζX3)
    return exp(k0 + x_0*(k1 + x_0*(k2 + x_0*k3)))
end

# ϕ: 7 tuples of 6 Float64 — GPU-safe constant (same as SAFTVRMieconsts.ϕ)
const SAFTVRMIE_PHI = (
    ( 7.5365557, -359.440,  1550.9, -1.199320, -1911.2800,  9236.9),
    (-37.604630,  1825.60, -5070.1,  9.063632,  21390.175,-129430.0),
    ( 71.745953, -3168.00,  6534.6,-17.94820,  -51320.700, 357230.0),
    (-46.835520,  1884.20, -3288.7, 11.34027,   37064.540,-315530.0),
    ( -2.4679820,  -0.82376,-2.7171, 20.52142,   1103.7420,  1390.2),
    ( -0.5027200,  -3.19350, 2.0883,-56.63770,  -3264.6100, -4518.2),
    (  8.0956883,   3.70900, 0.0000, 40.53683,   2556.1810,  4241.6),
)

@inline function _f123456_kernel(α, ϕ)
    fa1=0.0;fa2=0.0;fa3=0.0;fa4=0.0;fa5=0.0;fa6=0.0
    fb1=0.0;fb2=0.0;fb3=0.0;fb4=0.0;fb5=0.0;fb6=0.0
    αi = 1.0
    for i in 1:4
        p = ϕ[i]
        fa1+=p[1]*αi; fa2+=p[2]*αi; fa3+=p[3]*αi
        fa4+=p[4]*αi; fa5+=p[5]*αi; fa6+=p[6]*αi
        αi *= α
    end
    αi = α
    for i in 5:7
        p = ϕ[i]
        fb1+=p[1]*αi; fb2+=p[2]*αi; fb3+=p[3]*αi
        fb4+=p[4]*αi; fb5+=p[5]*αi; fb6+=p[6]*αi
        αi *= α
    end
    return (fa1/(1.0+fb1), fa2/(1.0+fb2), fa3/(1.0+fb3),
            fa4/(1.0+fb4), fa5/(1.0+fb5), fa6/(1.0+fb6))
end

"""
SAFT-VR Mie dispersion contribution at grid point `kk`.
- `meff[i]`: effective segment count (m*S for SAFTγMie, m otherwise)
- `IDX_ρz`: field index for the dispersion weighted density
  - 6+ND for SAFTVRMie and SAFTgammaMie
  - 4+ND for COFFEE
Used by SAFTVRMieModel, SAFTgammaMieModel, COFFEEModel.
"""
@inline function f_disp(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}, ::Val{IDX_ρz}) where {NC, ND, IDX_ρz, M}
    meff     = params.meff
    HSd      = params.HSd
    sigma    = params.sigma
    epsilon  = params.epsilon
    lambda_r = params.lambda_r_t
    lambda_a = params.lambda_a_t
    psi_eff  = params.psi_eff
    A        = params.A
    phi      = params.phi
    ρS_d = 0.0
    @inbounds for i in 1:NC
        ρS_d += n[kk, IDX_ρz, i] * 3.0/(4.0*π*psi_eff[i]^3) * meff[i]
    end
    kρS_d = ρS_d * π/6.0/8.0

    ζ_Xd=0.0;  σ3_xd=0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄zi  = n[kk, IDX_ρz, i] * 3.0/(4.0*π*psi_eff[i]^3)
        x_Si = ρ̄zi * meff[i] / ρS_d
        @inbounds for j in 1:NC
            dj   = HSd[j]
            ρ̄zj  = n[kk, IDX_ρz, j] * 3.0/(4.0*π*psi_eff[j]^3)
            x_Sj = ρ̄zj * meff[j] / ρS_d
            σ3_xd += x_Si*x_Sj*sigma[i,j]^3
            ζ_Xd  += kρS_d*x_Si*x_Sj*(di+dj)^3
        end
    end
    ζstd  = σ3_xd * ρS_d * π/6.0
    ζst5d = ζstd^5;  ζst8d = ζstd^8
    KHSd  = _KHS_kernel(ζ_Xd)

    a₁=0.0;  a₂=0.0;  a₃=0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄zi  = n[kk, IDX_ρz, i] * 3.0/(4.0*π*psi_eff[i]^3)
        x_Si = ρ̄zi * meff[i] / ρS_d
        λa=lambda_a[i][i]; λr=lambda_r[i][i]; σii=sigma[i,i]; ϵii=epsilon[i,i]
        _C=_Cλ_kernel(λa,λr);  x0=σii/di;  dij3=di^3
        aS1_a=_aS1_kernel(λa,     ζ_Xd,A); B_a=_B_kernel(λa,    x0,ζ_Xd)
        aS1_r=_aS1_kernel(λr,     ζ_Xd,A); B_r=_B_kernel(λr,    x0,ζ_Xd)
        a1ij = 2.0*π*ϵii*dij3*_C*ρS_d*(x0^λa*(aS1_a+B_a)-x0^λr*(aS1_r+B_r))
        aS1_2a=_aS1_kernel(2.0*λa, ζ_Xd,A); B_2a=_B_kernel(2.0*λa, x0,ζ_Xd)
        aS1_2r=_aS1_kernel(2.0*λr, ζ_Xd,A); B_2r=_B_kernel(2.0*λr, x0,ζ_Xd)
        aS1_ar=_aS1_kernel(λa+λr,  ζ_Xd,A); B_ar=_B_kernel(λa+λr,  x0,ζ_Xd)
        α=_C*(1.0/(λa-3.0)-1.0/(λr-3.0))
        f1,f2,f3,f4,f5,f6 = _f123456_kernel(α,phi)
        χ = f1*ζstd+f2*ζst5d+f3*ζst8d
        a2ij = π*KHSd*(1.0+χ)*ρS_d*ϵii^2*dij3*_C^2*(
               x0^(2.0*λa)*(aS1_2a+B_2a)
             - 2.0*x0^(λa+λr)*(aS1_ar+B_ar)
             + x0^(2.0*λr)*(aS1_2r+B_2r))
        a3ij = -ϵii^3*f4*ζstd*exp(f5*ζstd+f6*ζstd^2)
        a₁ += a1ij*x_Si*x_Si;  a₂ += a2ij*x_Si*x_Si;  a₃ += a3ij*x_Si*x_Si
        @inbounds for j in 1:NC
            if j != i
                dj   = HSd[j]
                ρ̄zj  = n[kk, IDX_ρz, j] * 3.0/(4.0*π*psi_eff[j]^3)
                x_Sj = ρ̄zj * meff[j] / ρS_d
                λa2=lambda_a[i][j]; λr2=lambda_r[i][j]; σij=sigma[i,j]; ϵij=epsilon[i,j]
                _C2=_Cλ_kernel(λa2,λr2); dij2=0.5*(di+dj); dij3_2=dij2^3; x0ij=σij/dij2
                aS1_a2=_aS1_kernel(λa2,      ζ_Xd,A); B_a2=_B_kernel(λa2,      x0ij,ζ_Xd)
                aS1_r2=_aS1_kernel(λr2,      ζ_Xd,A); B_r2=_B_kernel(λr2,      x0ij,ζ_Xd)
                a1ij2 = 2.0*π*ϵij*dij3_2*_C2*ρS_d*(x0ij^λa2*(aS1_a2+B_a2)-x0ij^λr2*(aS1_r2+B_r2))
                aS1_2a2=_aS1_kernel(2.0*λa2, ζ_Xd,A); B_2a2=_B_kernel(2.0*λa2, x0ij,ζ_Xd)
                aS1_2r2=_aS1_kernel(2.0*λr2, ζ_Xd,A); B_2r2=_B_kernel(2.0*λr2, x0ij,ζ_Xd)
                aS1_ar2=_aS1_kernel(λa2+λr2, ζ_Xd,A); B_ar2=_B_kernel(λa2+λr2, x0ij,ζ_Xd)
                α2=_C2*(1.0/(λa2-3.0)-1.0/(λr2-3.0))
                f1_2,f2_2,f3_2,f4_2,f5_2,f6_2 = _f123456_kernel(α2,phi)
                χ2 = f1_2*ζstd+f2_2*ζst5d+f3_2*ζst8d
                a2ij2 = π*KHSd*(1.0+χ2)*ρS_d*ϵij^2*dij3_2*_C2^2*(
                        x0ij^(2.0*λa2)*(aS1_2a2+B_2a2)
                      - 2.0*x0ij^(λa2+λr2)*(aS1_ar2+B_ar2)
                      + x0ij^(2.0*λr2)*(aS1_2r2+B_2r2))
                a3ij2 = -ϵij^3*f4_2*ζstd*exp(f5_2*ζstd+f6_2*ζstd^2)
                a₁ += a1ij2*x_Si*x_Sj
                a₂ += a2ij2*x_Si*x_Sj
                a₃ += a3ij2*x_Si*x_Sj
            end
        end
    end
    return ρS_d * (a₁/T + a₂/(T*T) + a₃/(T*T*T))
end

# SAFTVRMie-specific association strength: Δ = expm1(ε_assoc/T) * κ * I(Tr, ρr)
# where I is an 11×11 polynomial in (Tr=T/ε_Mie, ρr=ρS*σ³_x).
# Overrides the default g_hs _assoc_delta via more specific M <: SAFTVRMieModel dispatch.
@inline function _assoc_delta(p, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix,
                               ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTVRMieModel}
    p > n_pairs && return 0.0

    # Compute ρS from individual n₃ fields (field index 3 = F2+1)
    ρS = 0.0
    @inbounds for k in 1:NC
        ρS += n[kk, 3, k] * 6.0 / (π * params.HSd[k]^3) * params.m[k]
    end

    # σ³_x double loop: x_Sk = ρ̄k * m[k] / ρS
    σ3_x = 0.0
    @inbounds for k in 1:NC
        ρ̄k  = n[kk, 3, k] * 6.0 / (π * params.HSd[k]^3)
        xSk = ρ̄k * params.m[k] / ρS
        σ3_x += xSk * xSk * params.sigma[k,k]^3
        @inbounds for l in 1:(k-1)
            ρ̄l  = n[kk, 3, l] * 6.0 / (π * params.HSd[l]^3)
            xSl = ρ̄l * params.m[l] / ρS
            σ3_x += 2.0 * xSk * xSl * params.sigma[k,l]^3
        end
    end
    ρr = ρS * σ3_x

    ic = _nti(params.assoc_icomp, p)
    jc = _nti(params.assoc_jcomp, p)
    Tr = T / params.epsilon[ic, jc]

    # I(Tr, ρr): c stored as NTuple{11, NTuple{11, Float64}} (row = n-index, col = m-index)
    I_val = 0.0
    ρrn   = 1.0
    for ni in 0:10
        row = _nti(params.VRMie_c, ni + 1)
        Trm = 1.0
        for mi in 0:10
            I_val += _nti(row, mi + 1) * Trm * ρrn
            Trm *= Tr
        end
        ρrn *= ρr
    end

    return expm1(params.assoc_eps[p] / T) * params.assoc_kap[p] * I_val
end

function preallocate_params(system::DFTSystem{<:SAFTVRMieModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model
    nc = length(model)
    lr = model.params.lambda_r.values
    la = model.params.lambda_a.values
    lambda_r_t = ntuple(i -> ntuple(j -> lr[i,j], nc), nc)
    lambda_a_t = ntuple(i -> ntuple(j -> la[i,j], nc), nc)
    base = (;
        HSd        = adapt_to_device(backend, FP, system.species.size),
        m          = adapt_to_device(backend, FP, model.params.segment.values),
        meff       = adapt_to_device(backend, FP, model.params.segment.values),
        sigma      = adapt_to_device(backend, FP, model.params.sigma.values),
        epsilon    = adapt_to_device(backend, FP, model.params.epsilon.values),
        lambda_r_t = lambda_r_t,
        lambda_a_t = lambda_a_t,
        psi_eff    = adapt_to_device(backend, FP, system.fields[end].width),
        A          = SAFTVRMIE_A,
        phi        = SAFTVRMIE_PHI,
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params(model, system.species.size)

        nc_model         = length(model)
        ia_global_v      = [n_sites_cumsum_v[assoc_icomp_v[p]] + assoc_isite_v[p] for p in 1:nn]
        jb_global_v      = [n_sites_cumsum_v[assoc_jcomp_v[p]] + assoc_jsite_v[p] for p in 1:nn]
        n_ia_v           = [n_sites_flat_v[ia_global_v[p]] for p in 1:nn]
        n_jb_v           = [n_sites_flat_v[jb_global_v[p]] for p in 1:nn]
        assoc_icomp_t    = ntuple(p -> assoc_icomp_v[p],    Val(nn))
        assoc_jcomp_t    = ntuple(p -> assoc_jcomp_v[p],    Val(nn))
        assoc_isite_t    = ntuple(p -> assoc_isite_v[p],    Val(nn))
        assoc_jsite_t    = ntuple(p -> assoc_jsite_v[p],    Val(nn))
        assoc_ia_global_t = ntuple(p -> ia_global_v[p],     Val(nn))
        assoc_jb_global_t = ntuple(p -> jb_global_v[p],     Val(nn))
        assoc_n_ia_t      = ntuple(p -> n_ia_v[p],          Val(nn))
        assoc_n_jb_t      = ntuple(p -> n_jb_v[p],          Val(nn))
        n_sites_flat_t   = ntuple(j -> n_sites_flat_v[j],   Val(total_sites))
        n_sites_cumsum_t = ntuple(i -> n_sites_cumsum_v[i], Val(nc_model + 1))

        # Pack I(Tr,ρr) polynomial as NTuple{11,NTuple{11,Float64}}: row=n-index, col=m-index
        c_mat   = SAFTVRMieconsts.c
        VRMie_c = ntuple(ni -> ntuple(mi -> c_mat[ni, mi], 11), 11)

        assoc = (;
            has_assoc       = true,
            assoc_n_pairs   = Val(nn),
            assoc_n_sites   = Val(total_sites),
            assoc_icomp     = assoc_icomp_t,
            assoc_jcomp     = assoc_jcomp_t,
            assoc_isite     = assoc_isite_t,
            assoc_jsite     = assoc_jsite_t,
            assoc_ia_global = assoc_ia_global_t,
            assoc_jb_global = assoc_jb_global_t,
            assoc_n_ia      = assoc_n_ia_t,
            assoc_n_jb      = assoc_n_jb_t,
            assoc_eps       = adapt_to_device(backend, FP, assoc_eps_v),
            assoc_kap       = adapt_to_device(backend, FP, assoc_kap_v),
            assoc_sig3      = adapt_to_device(backend, FP, assoc_sig3_v),
            assoc_dij       = adapt_to_device(backend, FP, assoc_dij_v),
            n_sites_flat    = n_sites_flat_t,
            n_sites_cumsum  = n_sites_cumsum_t,
            total_sites,
            VRMie_c,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    return params, length(model)
end
