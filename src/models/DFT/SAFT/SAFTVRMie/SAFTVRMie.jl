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

function get_fields(model::SAFTVRMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    nc = length(model)
    ngrid = structure.ngrid
    ω = structure_ω(structure, device)
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
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk
    size = d(model,1e-3,T,ρbulk)

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T
    nc = length(model)
    return SAFTVRMieSpecies(ones(Int64,nc),size,ρbulk,μres)
end

function get_propagator(model::SAFTVRMieModel, species::DFTSpecies, structure::DFTStructure)
    return IdealPropagator()
end

function f_res(system::Union{DFTSystem,ElectrolyteDFTSystem}, model::SAFTVRMieModel, n)
    nd = dimension(system)
    n1,n2,n3,n4,n5,n6,n7 = @view(n[1,:]),@view(n[2,:]),@view(n[3,:]),@view(n[4:4+nd-1,:]),@view(n[4+nd,:]),@view(n[5+nd,:]),@view(n[6+nd,:])
    return f_hs(system,model,n2,n3,n4) + f_chain(system,model,n1,n5,n6) + f_disp(system,model,n7) + f_assoc(system,model,n2,n3,n4)
end

function f_chain(system::Union{DFTSystem,ElectrolyteDFTSystem}, model::SAFTVRMieModel, ρhc, ρ̄hc, _λ)
    V = nothing
    T = system.structure.conditions[2]
    _d = system.species.size
    m = model.params.segment
    _ϵ = model.params.epsilon
    _λr = model.params.lambda_r
    _λa = model.params.lambda_a
    _σ = model.params.sigma

    ρ̄hc = ρ̄hc*3 ./(4 .*_d.^3)/π

    z = ρ̄hc /sum(ρ̄hc)
    m̄ = dot(z,m)
    m̄inv = 1/m̄

    ρS = dot(ρ̄hc,m)

    _ζ_X = zero(T+first(ρ̄hc)+one(eltype(model)))
    kρS = ρS* π/6/8
    σ3_x = _ζ_X

    for i ∈ @comps
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        di =_d[i]
        r1 = kρS*x_Si*x_Si*(2*di)^3
        _ζ_X += r1
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3_x += 2*x_Si*x_Sj*(_σ[i,j]^3)
            dij = (di + _d[j])
            r1 = kρS*x_Si*x_Sj*dij^3
            _ζ_X += 2*r1
        end
    end

    _ζst = σ3_x*ρS*π/6

    fchain = zero(T+first(z)+one(eltype(model)))
    _KHS,_∂KHS = @f(KHS_fdf,_ζ_X,ρS)
    for i ∈ @comps
        ϵ = _ϵ[i,i]
        λa = _λa[i,i]
        λr = _λr[i,i]
        σ = _σ[i,i]
        _C = @f(Cλ,λa,λr)
        dij = _d[i]
        x_0ij = σ/dij
        x_0ij = σ/dij
        #calculations for a1 - diagonal
        aS_1_a,∂aS_1∂ρS_a = @f(aS_1_fdf,λa,_ζ_X,ρS)
        aS_1_r,∂aS_1∂ρS_r = @f(aS_1_fdf,λr,_ζ_X,ρS)
        B_a,∂B∂ρS_a = @f(B_fdf,λa,x_0ij,_ζ_X,ρS)
        B_r,∂B∂ρS_r = @f(B_fdf,λr,x_0ij,_ζ_X,ρS)

        #calculations for a2 - diagonal
        aS_1_2a,∂aS_1∂ρS_2a = @f(aS_1_fdf,2*λa,_ζ_X,ρS)
        aS_1_2r,∂aS_1∂ρS_2r = @f(aS_1_fdf,2*λr,_ζ_X,ρS)
        aS_1_ar,∂aS_1∂ρS_ar = @f(aS_1_fdf,λa+λr,_ζ_X,ρS)
        B_2a,∂B∂ρS_2a = @f(B_fdf,2*λa,x_0ij,_ζ_X,ρS)
        B_2r,∂B∂ρS_2r = @f(B_fdf,2*λr,x_0ij,_ζ_X,ρS)
        B_ar,∂B∂ρS_ar = @f(B_fdf,λr+λa,x_0ij,_ζ_X,ρS)
        α = _C*(1/(λa-3)-1/(λr-3))
        g_HSi = @f(g_HS,x_0ij,_ζ_X)
        #@show (g_HSi,i)
        ∂a_1∂ρ_S = _C*(x_0ij^λa*(∂aS_1∂ρS_a+∂B∂ρS_a)
                      - x_0ij^λr*(∂aS_1∂ρS_r+∂B∂ρS_r))
        #@show (∂a_1∂ρ_S,1)

        g_1_ = 3*∂a_1∂ρ_S-_C*(λa*x_0ij^λa*(aS_1_a+B_a)-λr*x_0ij^λr*(aS_1_r+B_r))
        #@show (g_1_,i)
        θ = exp(ϵ/T)-1
        γc = 10 * (-tanh(10*(0.57-α))+1) * _ζst*θ*exp(-6.7*_ζst-8*_ζst^2)
        ∂a_2∂ρ_S = 0.5*_C^2 *
            (ρS*_∂KHS*(x_0ij^(2*λa)*(aS_1_2a+B_2a)
            - 2*x_0ij^(λa+λr)*(aS_1_ar+B_ar)
            + x_0ij^(2*λr)*(aS_1_2r+B_2r))
            + _KHS*(x_0ij^(2*λa)*(∂aS_1∂ρS_2a+∂B∂ρS_2a)
            - 2*x_0ij^(λa+λr)*(∂aS_1∂ρS_ar+∂B∂ρS_ar)
            + x_0ij^(2*λr)*(∂aS_1∂ρS_2r+∂B∂ρS_2r)))

        gMCA2 = 3*∂a_2∂ρ_S-_KHS*_C^2 *
        (λr*x_0ij^(2*λr)*(aS_1_2r+B_2r)-
            (λa+λr)*x_0ij^(λa+λr)*(aS_1_ar+B_ar)+
            λa*x_0ij^(2*λa)*(aS_1_2a+B_2a))
        g_2_ = (1+γc)*gMCA2
        #@show (g_2_,i)
        g_Mie_ = g_HSi*exp(ϵ/T*g_1_/g_HSi+(ϵ/T)^2*g_2_/g_HSi)
        #@show (g_Mie_,i)
        λ = _λ[i]/(2*_d[i])
        fchain +=  ρhc[i]*(log(g_Mie_*λ/ρhc[i])*(m[i]-1))
    end
    
    return -fchain
end

function f_disp(system::Union{DFTSystem,ElectrolyteDFTSystem}, model::SAFTVRMieModel, ρ̄)
    V = nothing
    ψ = system.fields[end].width
    _d = system.species.size
    T = system.structure.conditions[2]
    m = model.params.segment
    _ϵ = model.params.epsilon
    _λr = model.params.lambda_r
    _λa = model.params.lambda_a
    _σ = model.params.sigma

    ρ̄ = ρ̄*3 ./(4*ψ.^3)/π
    ∑ρ̄ = sum(ρ̄)
    z = ρ̄ /∑ρ̄
    m̄ = dot(z,m)
    m̄inv = 1/m̄
    ∑z = sum(z)

    ρS = dot(ρ̄,m)

    _ζ_X = zero(T+first(ρ̄)+one(eltype(model)))
    kρS = ρS* π/6/8
    σ3_x = _ζ_X

    for i ∈ @comps
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        di =_d[i]
        r1 = kρS*x_Si*x_Si*(2*di)^3
        _ζ_X += r1
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3_x += 2*x_Si*x_Sj*(_σ[i,j]^3)
            dij = (di + _d[j])
            r1 = kρS*x_Si*x_Sj*dij^3
            _ζ_X += 2*r1
        end
    end

    _ζst = σ3_x*ρS*π/6
    
    a₁ = zero(T+first(z)+one(eltype(model)))
    a₂ = a₁
    a₃ = a₁
    _ζst5 = _ζst^5
    _ζst8 = _ζst^8
    _KHS = @f(KHS,_ζ_X,ρS)
    for i ∈ @comps
        j = i
        x_Si = z[i]*m[i]*m̄inv
        x_Sj = x_Si
        ϵ = _ϵ[i,j]
        λa = _λa[i,i]
        λr = _λr[i,i]
        σ = _σ[i,i]
        _C = @f(Cλ,λa,λr)
        dij = _d[i]
        dij3 = dij^3
        x_0ij = σ/dij
        #calculations for a1 - diagonal
        aS_1_a = @f(aS_1,λa,_ζ_X)
        aS_1_r = @f(aS_1,λr,_ζ_X)
        B_a = @f(B,λa,x_0ij,_ζ_X)
        B_r = @f(B,λr,x_0ij,_ζ_X)
        a1_ij = (2*π*ϵ*dij3)*_C*ρS*
        (x_0ij^λa*(aS_1_a+B_a) - x_0ij^λr*(aS_1_r+B_r))

        #calculations for a2 - diagonal
        aS_1_2a = @f(aS_1,2*λa,_ζ_X)
        aS_1_2r = @f(aS_1,2*λr,_ζ_X)
        aS_1_ar = @f(aS_1,λa+λr,_ζ_X)
        B_2a = @f(B,2*λa,x_0ij,_ζ_X)
        B_2r = @f(B,2*λr,x_0ij,_ζ_X)
        B_ar = @f(B,λr+λa,x_0ij,_ζ_X)
        α = _C*(1/(λa-3)-1/(λr-3))
        f1,f2,f3,f4,f5,f6 = @f(f123456,α)
        _χ = f1*_ζst+f2*_ζst5+f3*_ζst8
        a2_ij = π*_KHS*(1+_χ)*ρS*ϵ^2*dij3*_C^2 *
        (x_0ij^(2*λa)*(aS_1_2a+B_2a)
        - 2*x_0ij^(λa+λr)*(aS_1_ar+B_ar)
        + x_0ij^(2*λr)*(aS_1_2r+B_2r))

        #calculations for a3 - diagonal
        a3_ij = -ϵ^3*f4*_ζst * exp(f5*_ζst+f6*_ζst^2)
        #adding - diagonal
        a₁ += a1_ij*x_Si*x_Si
        a₂ += a2_ij*x_Si*x_Si
        a₃ += a3_ij*x_Si*x_Si
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            ϵ = _ϵ[i,j]
            λa = _λa[i,j]
            λr = _λr[i,j]
            σ = _σ[i,j]
            _C = @f(Cλ,λa,λr)
            dij = 0.5*(_d[i]+_d[j])
            x_0ij = σ/dij
            dij3 = dij^3
            x_0ij = σ/dij
            #calculations for a1
            a1_ij = (2*π*ϵ*dij3)*_C*ρS*
            (x_0ij^λa*(@f(aS_1,λa,_ζ_X)+@f(B,λa,x_0ij,_ζ_X)) - x_0ij^λr*(@f(aS_1,λr,_ζ_X)+@f(B,λr,x_0ij,_ζ_X)))

            #calculations for a2
            α = _C*(1/(λa-3)-1/(λr-3))
            f1,f2,f3,f4,f5,f6 = @f(f123456,α)
            _χ = f1*_ζst+f2*_ζst5+f3*_ζst8
            a2_ij = π*_KHS*(1+_χ)*ρS*ϵ^2*dij3*_C^2 *
            (x_0ij^(2*λa)*(@f(aS_1,2*λa,_ζ_X)+@f(B,2*λa,x_0ij,_ζ_X))
            - 2*x_0ij^(λa+λr)*(@f(aS_1,λa+λr,_ζ_X)+@f(B,λa+λr,x_0ij,_ζ_X))
            + x_0ij^(2*λr)*(@f(aS_1,2λr,_ζ_X)+@f(B,2*λr,x_0ij,_ζ_X)))

            #calculations for a3
            a3_ij = -ϵ^3*f4*_ζst * exp(f5*_ζst+f6*_ζst^2)
            #adding
            a₁ += 2*a1_ij*x_Si*x_Sj
            a₂ += 2*a2_ij*x_Si*x_Sj
            a₃ += 2*a3_ij*x_Si*x_Sj
        end
    end
    a₁ = a₁*m̄/T/∑z #/sum(z)
    a₂ = a₂*m̄/(T*T)/∑z  #/sum(z)
    a₃ = a₃*m̄/(T*T*T)/∑z  #/sum(z)
    #@show (a₁,a₂,a₃)
    adisp = a₁ + a₂ + a₃
    return ∑ρ̄*adisp
end

function Δ(model::SAFTVRMieModel, T, n, n₃, nᵥ, i, j, a, b)
    _d = d(model,1e-3,T,onevec(model))
    _σ = model.params.sigma.values
    m = model.params.segment.values
    ϵ_assoc = model.params.epsilon_assoc.values
    K = model.params.bondvol.values[i,j][a,b]
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(K))
    iszero(K) && return _0

    ρ̄ = n₃*3*2 ./(_d.^3)/π

    z = ρ̄ /sum(ρ̄)
    m̄ = dot(z,m)
    m̄inv = 1/m̄

    ρS = dot(ρ̄,m)

    σ3_x = zero(T+first(z)+one(eltype(model)))

    for i ∈ @comps
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3_x += 2*x_Si*x_Sj*(_σ[i,j]^3)
        end
    end

    ρr  = ρS*σ3_x
    
    ϵ = model.params.epsilon
    Tr = T/ϵ[i,j]
    _I = I(model,Tr,ρr)
    
    F = expm1(ϵ_assoc[i,j][a,b]/T)

    return F*K*_I
end

function I(model::SAFTVRMieModel, Tr,ρr)
    c  = SAFTVRMieconsts.c
    res = zero(ρr+Tr)
    @inbounds for n ∈ 0:10
        ρrn = ρr^n
        res_m = zero(res)
        for m ∈ 0:(10-n)
            res_m += c[n+1,m+1]*Tr^m
        end
        res += res_m*ρrn
    end
    return res
end
# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────
# GPU-safe constants: row-tuple form of the SAFTγMie A matrix for ζeff.
const SAFTVRMIE_A = (
    ( 0.81096,    1.7888,   -37.578,   92.284),
    ( 1.02050,  -19.341,   151.26,   -463.50),
    (-1.90570,   22.845,  -228.14,   973.92),
    ( 1.08850,   -6.1962,  106.98,  -677.64),
)

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
    return f, ρdf / (ρS + 1e-15)
end

@inline function _gHS_kernel(x_0, ζ_X)
    ζX3 = (1.0-ζ_X)^3
    k0 = -Base.log(1.0-ζ_X) + evalpoly(ζ_X,(0.0,42.0,-39.0,9.0,-2.0))/(6.0*ζX3)
    k1 = evalpoly(ζ_X,(0.0,-12.0,6.0,0.0,1.0)) / (2.0*ζX3)
    k2 = -3.0*ζ_X*ζ_X / (8.0*(1.0-ζ_X)^2)
    k3 = evalpoly(ζ_X,(0.0,3.0,3.0,0.0,-1.0)) / (6.0*ζX3)
    return exp(evalpoly(x_0,(k0,k1,k2,k3)))
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
    f_disp(n, meff, HSd, sigma, epsilon, lambda_r, lambda_a, psi_eff,
           kk, T, ::Val{NC}, ::Val{ND}, ::Val{IDX_ρz}, A, phi, ::Type{M})

SAFT-VR Mie dispersion contribution at grid point `kk`.
- `meff[i]`: effective segment count (m*S for SAFTγMie, m otherwise)
- `IDX_ρz`: field index for the dispersion weighted density
  - 6+ND for SAFTVRMie and SAFTgammaMie
  - 4+ND for COFFEE
Used by SAFTVRMieModel, SAFTgammaMieModel, COFFEEModel.
"""
@inline function f_disp(n, meff, HSd, sigma, epsilon, lambda_r, lambda_a, psi_eff,
                        kk, T, ::Val{NC}, ::Val{ND}, ::Val{IDX_ρz}, A, phi, ::Type{M}) where {NC, ND, IDX_ρz, M}
    _pi   = 3.141592653589793
    eps_v = 1e-15

    ρS_d = eps_v
    @inbounds for i in 1:NC
        ρS_d += n[kk, IDX_ρz, i] * 3.0/(4.0*_pi*psi_eff[i]^3) * meff[i]
    end
    kρS_d = ρS_d * _pi/6.0/8.0

    ζ_Xd=0.0;  σ3_xd=0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄zi  = n[kk, IDX_ρz, i] * 3.0/(4.0*_pi*psi_eff[i]^3)
        x_Si = ρ̄zi * meff[i] / ρS_d
        σ3_xd += x_Si*x_Si*sigma[i,i]^3
        ζ_Xd  += kρS_d*x_Si*x_Si*(2.0*di)^3
        @inbounds for j in 1:(i-1)
            dj   = HSd[j]
            ρ̄zj  = n[kk, IDX_ρz, j] * 3.0/(4.0*_pi*psi_eff[j]^3)
            x_Sj = ρ̄zj * meff[j] / ρS_d
            σ3_xd += 2.0*x_Si*x_Sj*sigma[i,j]^3
            ζ_Xd  += 2.0*kρS_d*x_Si*x_Sj*(di+dj)^3
        end
    end
    ζstd  = σ3_xd * ρS_d * _pi/6.0
    ζst5d = ζstd^5;  ζst8d = ζstd^8
    KHSd  = _KHS_kernel(ζ_Xd)

    a₁=0.0;  a₂=0.0;  a₃=0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄zi  = n[kk, IDX_ρz, i] * 3.0/(4.0*_pi*psi_eff[i]^3)
        x_Si = ρ̄zi * meff[i] / ρS_d
        λa=lambda_a[i,i]; λr=lambda_r[i,i]; σii=sigma[i,i]; ϵii=epsilon[i,i]
        _C=_Cλ_kernel(λa,λr);  x0=σii/di;  dij3=di^3
        aS1_a=_aS1_kernel(λa,     ζ_Xd,A); B_a=_B_kernel(λa,    x0,ζ_Xd)
        aS1_r=_aS1_kernel(λr,     ζ_Xd,A); B_r=_B_kernel(λr,    x0,ζ_Xd)
        a1ij = 2.0*_pi*ϵii*dij3*_C*ρS_d*(x0^λa*(aS1_a+B_a)-x0^λr*(aS1_r+B_r))
        aS1_2a=_aS1_kernel(2.0*λa, ζ_Xd,A); B_2a=_B_kernel(2.0*λa, x0,ζ_Xd)
        aS1_2r=_aS1_kernel(2.0*λr, ζ_Xd,A); B_2r=_B_kernel(2.0*λr, x0,ζ_Xd)
        aS1_ar=_aS1_kernel(λa+λr,  ζ_Xd,A); B_ar=_B_kernel(λa+λr,  x0,ζ_Xd)
        α=_C*(1.0/(λa-3.0)-1.0/(λr-3.0))
        f1,f2,f3,f4,f5,f6 = _f123456_kernel(α,phi)
        χ = f1*ζstd+f2*ζst5d+f3*ζst8d
        a2ij = _pi*KHSd*(1.0+χ)*ρS_d*ϵii^2*dij3*_C^2*(
               x0^(2.0*λa)*(aS1_2a+B_2a)
             - 2.0*x0^(λa+λr)*(aS1_ar+B_ar)
             + x0^(2.0*λr)*(aS1_2r+B_2r))
        a3ij = -ϵii^3*f4*ζstd*exp(f5*ζstd+f6*ζstd^2)
        a₁ += a1ij*x_Si*x_Si;  a₂ += a2ij*x_Si*x_Si;  a₃ += a3ij*x_Si*x_Si
        @inbounds for j in 1:(i-1)
            dj   = HSd[j]
            ρ̄zj  = n[kk, IDX_ρz, j] * 3.0/(4.0*_pi*psi_eff[j]^3)
            x_Sj = ρ̄zj * meff[j] / ρS_d
            λa2=lambda_a[i,j]; λr2=lambda_r[i,j]; σij=sigma[i,j]; ϵij=epsilon[i,j]
            _C2=_Cλ_kernel(λa2,λr2); dij2=0.5*(di+dj); dij3_2=dij2^3; x0ij=σij/dij2
            aS1_a2=_aS1_kernel(λa2,      ζ_Xd,A); B_a2=_B_kernel(λa2,      x0ij,ζ_Xd)
            aS1_r2=_aS1_kernel(λr2,      ζ_Xd,A); B_r2=_B_kernel(λr2,      x0ij,ζ_Xd)
            a1ij2 = 2.0*_pi*ϵij*dij3_2*_C2*ρS_d*(x0ij^λa2*(aS1_a2+B_a2)-x0ij^λr2*(aS1_r2+B_r2))
            aS1_2a2=_aS1_kernel(2.0*λa2, ζ_Xd,A); B_2a2=_B_kernel(2.0*λa2, x0ij,ζ_Xd)
            aS1_2r2=_aS1_kernel(2.0*λr2, ζ_Xd,A); B_2r2=_B_kernel(2.0*λr2, x0ij,ζ_Xd)
            aS1_ar2=_aS1_kernel(λa2+λr2, ζ_Xd,A); B_ar2=_B_kernel(λa2+λr2, x0ij,ζ_Xd)
            α2=_C2*(1.0/(λa2-3.0)-1.0/(λr2-3.0))
            f1_2,f2_2,f3_2,f4_2,f5_2,f6_2 = _f123456_kernel(α2,phi)
            χ2 = f1_2*ζstd+f2_2*ζst5d+f3_2*ζst8d
            a2ij2 = _pi*KHSd*(1.0+χ2)*ρS_d*ϵij^2*dij3_2*_C2^2*(
                    x0ij^(2.0*λa2)*(aS1_2a2+B_2a2)
                  - 2.0*x0ij^(λa2+λr2)*(aS1_ar2+B_ar2)
                  + x0ij^(2.0*λr2)*(aS1_2r2+B_2r2))
            a3ij2 = -ϵij^3*f4_2*ζstd*exp(f5_2*ζstd+f6_2*ζstd^2)
            a₁ += 2.0*a1ij2*x_Si*x_Sj
            a₂ += 2.0*a2ij2*x_Si*x_Sj
            a₃ += 2.0*a3ij2*x_Si*x_Sj
        end
    end
    return ρS_d * (a₁/T + a₂/(T*T) + a₃/(T*T*T))
end

"""
SAFT-VR Mie chain contribution (gMie contact value) at grid point `kk`.
"""
@inline function f_chain(n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTVRMieModel}
    _pi   = 3.141592653589793
    eps_v = 1e-15

    HSd     = params.HSd
    m_seg   = params.m
    σ       = params.sigma
    ϵ       = params.epsilon
    λr_mat  = params.lambda_r
    λa_mat  = params.lambda_a
    A       = params.A
    ϕ       = params.phi

    idx_ζ = 4+ND;  idx_λ = 5+ND

    ρS_c = eps_v
    @inbounds for i in 1:NC
        ρS_c += n[kk,idx_ζ,i] * 3.0/(4.0*_pi*HSd[i]^3) * m_seg[i]
    end
    kρS_c = ρS_c * _pi/6.0/8.0

    ζ_Xc = 0.0;  σ3_xc = 0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄hci = n[kk,idx_ζ,i] * 3.0/(4.0*_pi*di^3)
        x_Si = ρ̄hci * m_seg[i] / ρS_c
        σ3_xc += x_Si*x_Si*σ[i,i]^3
        ζ_Xc  += kρS_c * x_Si*x_Si*(2.0*di)^3
        @inbounds for j in 1:(i-1)
            dj   = HSd[j]
            ρ̄hcj = n[kk,idx_ζ,j] * 3.0/(4.0*_pi*dj^3)
            x_Sj = ρ̄hcj * m_seg[j] / ρS_c
            σ3_xc += 2.0*x_Si*x_Sj*σ[i,j]^3
            dij   = di+dj
            ζ_Xc  += 2.0*kρS_c*x_Si*x_Sj*dij^3
        end
    end
    ζstc = σ3_xc * ρS_c * _pi/6.0

    _KHSc, _∂KHSc = _KHS_fdf_kernel(ρS_c, ζ_Xc)

    res_chain = 0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄hci = n[kk,idx_ζ,i] * 3.0/(4.0*_pi*di^3)
        x_Si = ρ̄hci * m_seg[i] / ρS_c

        λa = λa_mat[i,i];  λr = λr_mat[i,i]
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
        res_chain += ρhci * Base.log(abs(gMie*λfld/(ρhci+eps_v))+eps_v) * (m_seg[i]-1.0)
    end
    return -res_chain
end

"""
Pointwise residual free energy for SAFT-VR Mie: FMT hard-sphere + chain + dispersion.

Field layout (same as PCSAFTModel):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc  (chain)
  5+ND     : ∫ρdz  with d    → λ    (chain)
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z   (dispersion)
"""
@inline function f_res(out, n, params, T, kk,
                       ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTVRMieModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_chain = f_chain(n, params, T, kk, Val(NC), Val(ND), M)
    res_disp  = f_disp(n, params.m, params.HSd, params.sigma, params.epsilon,
                       params.lambda_r, params.lambda_a, params.psi_eff,
                       kk, T, Val(NC), Val(ND), Val(6+ND), params.A, params.phi, M)
    out[kk] = res_hs + res_chain + res_disp
    return nothing
end

function preallocate_params(system::DFTSystem{<:SAFTVRMieModel})
    backend = system.options.device
    params = (;
        HSd      = Adapt.adapt(backend, system.species.size),
        m        = Adapt.adapt(backend, system.model.params.segment.values),
        sigma    = Adapt.adapt(backend, system.model.params.sigma.values),
        epsilon  = Adapt.adapt(backend, system.model.params.epsilon.values),
        lambda_r = Adapt.adapt(backend, system.model.params.lambda_r.values),
        lambda_a = Adapt.adapt(backend, system.model.params.lambda_a.values),
        psi_eff  = Adapt.adapt(backend, system.fields[end].width),
        A        = SAFTVRMIE_A,
        phi      = SAFTVRMIE_PHI,
    )
    nc = length(system.model)
    return params, nc
end
