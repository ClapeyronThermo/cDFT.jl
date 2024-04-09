using Clapeyron: SAFTVRMieModel
using Clapeyron: aS_1, B, KHS, Cλ, f123456
using Clapeyron: SAFTVRMieconsts

function F_res(model::SAFTVRMieModel,ρ,T,z)
    ψ = 1.3862
    HSd = d(model,1e-3,T,onevec(model))
    dz = ρ[1].mesh_size

    (n, n₃,nᵥ)  = weights_hs(model,ρ,z,1/2*HSd)
    (λ, ρ̄hc,_)    = weights_hs(model,ρ,z,HSd)
    (_, ρ̄,_)    = weights_hs(model,ρ,z,ψ*HSd)
    ρhc = zeros(length(z),length(ρ))
    for i in @comps
        ρhc[:,i] = ρ[i].density*N_A
    end

    nc = length(model)
    idx = 1:nc

    f1(x) = f_hs(model,T,@view(x[idx]),@view(x[idx.+nc]),@view(x[idx.+2*nc])
)+f_assoc(model,T,@view(x[idx]),@view(x[idx.+nc]),@view(x[idx.+2*nc])
)
    Φ_hs_assoc = mapslices(f1,[n n₃ nᵥ];dims=2)

#     f2(x) = f_chain(model,T,@view(x[idx]),@view(x[idx.+nc]),@view(x[idx.+2*nc])
# )
#     Φ_chain = mapslices(f2,[ρhc ρ̄hc λ];dims=2)
    
    f3(x) = f_disp(model,T,@view(x[idx]))
    Φ_disp = mapslices(f3,ρ̄;dims=2)
    
    Φ = Φ_disp+Φ_hs_assoc#+Φ_chain

    return ∫(Φ,dz)
end

function δFδρ_res(model::SAFTVRMieModel,ρ,T,z)
    return δFδρ_hs(model,ρ,T,z)+
           δFδρ_hc(model,ρ,T,z)+
           δFδρ_disp(model,ρ,T,z)+
           δFδρ_assoc(model,ρ,T,z)
end

function δFδρ_hc(model::SAFTVRMieModel,ρ,T,z)
    HSd = d(model,1e-3,T,onevec(model))
    lim = HSd

    (λ, ρ̄hc,_)  = weights_hs(model,ρ,z,lim)
    ρhc = zeros(length(z),length(ρ))
    for i in @comps
        ρhc[:,i] = ρ[i].density*N_A
    end

    nc = length(model)
    idx = 1:nc
    f(x) = f_hc(model,T,@view(x[idx]),@view(x[idx.+nc]),@view(x[idx.+2*nc]))
    df(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(df,[ρhc ρ̄hc λ];dims=2)
    ∂f∂ρhc0 = δfδn[:,idx]
    ∂f∂ρ̄hc0 = δfδn[:,idx.+nc]
    ∂f∂λ0 = δfδn[:,idx.+2*nc]

    δFδρ_hc = zeros(length(z),length(model))
    for i in @comps 
        bounds = ρ[i].bounds.+(-lim[i],lim[i])
        ∂f∂ρhc = DensityProfile(@view(∂f∂ρhc0[:,i]),z,bounds,[∂f∂ρhc0[1,i],∂f∂ρhc0[end,i]])
        ∂f∂ρ̄hc = DensityProfile(@view(∂f∂ρ̄hc0[:,i]),z,bounds,[∂f∂ρ̄hc0[1,i],∂f∂ρ̄hc0[end,i]])
        ∂f∂λ = DensityProfile(@view(∂f∂λ0[:,i]),z,bounds,[∂f∂λ0[1,i],∂f∂λ0[end,i]])
    
        span = range(-lim[i],lim[i],length=101)

        δFδρ_hc_1 = ∫ρdz.(Ref(∂f∂λ),z,Ref(span))
        δFδρ_hc_2 = π*∫ρz²dz.(Ref(∂f∂ρ̄hc),z,Ref(span))
        δFδρ_hc_3 = ∂f∂ρhc.(z)

        δFδρ_hc[:,i] = δFδρ_hc_1+δFδρ_hc_2+δFδρ_hc_3
    end
    return δFδρ_hc
end

function δFδρ_disp(model::SAFTVRMieModel,ρ,T,z)
    HSd = d(model,1e-3,T,onevec(model))
    lim = 1.3862*HSd

    (_, ρ̄,_)  = weights_hs(model,ρ,z,lim)

    nc = length(model)
    idx = 1:nc
    f(x) = f_disp(model,T,@view(x[idx]))
    df(x) = ForwardDiff.gradient(f,x)

    δfδn0  = mapslices(df,ρ̄;dims=2)
    ∂f∂n0 = δfδn0[:,idx]

    δFδρ_disp = zeros(length(z),length(model))
    for i in @comps 
        bounds = ρ[i].bounds.+(-lim[i],lim[i])
        ∂f∂n =  DensityProfile(∂f∂n0[:,i],z,bounds,[∂f∂n0[1,i],∂f∂n0[end,i]])
    
        span = range(-lim[i],lim[i],length=101) # Length = 101? Is it because len(z) = 101?

        δFδρ_disp[:,i] = π*∫ρz²dz.(Ref(∂f∂n),z,Ref(span))
    end

    return δFδρ_disp
end

function f_hc(model::SAFTVRMieModel, T, ρhc, ρ̄hc, _λ)
    HSd = d(model,1e-3,T,onevec(model))
    m = model.params.segment.values
    ζ₃ = zero(eltype(HSd)) + zero(eltype(ρ̄hc))
    ζ₂ = zero(ζ₃)
    for i in @comps
        mi,ρ̄hci,HSdi = m[i],ρ̄hc[i],HSd[i]
        ζ₃ += mi*ρ̄hci
        ζ₂ += mi*ρ̄hci/HSdi
    end
    ζ₃ *= 0.125
    ζ₂ *= 0.125
    #ζ₃ = 1/8*dot(m,ρ̄hc)
    #ζ₂ = sum(1/8*m.*ρ̄hc./HSd)
    ∑f = zero(ζ₃)
    for i in @comps
        λ = _λ[i]/(2*HSd[i])
        yᵈᵈ = 1/(1-ζ₃) + 1.5*HSd[i]*ζ₂/(1-ζ₃)^2+0.5*HSd[i]^2*ζ₂^2/(1-ζ₃)^3
        fi = -ρhc[i]*(m[i]-1)*log(yᵈᵈ*λ/ρhc[i])
        ∑f += fi
    end
    
    return ∑f
    #λ = _λ./(2*HSd) 
    #yᵈᵈ = @. 1/(1-ζ₃)+1.5*HSd*ζ₂/(1-ζ₃)^2+0.5*HSd^2*ζ₂^2/(1-ζ₃)^3
    #f = @. -ρhc*(m-1)*log(yᵈᵈ*λ/ρhc)
    #return sum(f)
end

function f_disp(model::SAFTVRMieModel, T, ρ̄)
    V = 1e-3
    ψ = 1.3862
    _d = d(model,V,T,onevec(model))
    m = model.params.segment
    _ϵ = model.params.epsilon
    _λr = model.params.lambda_r
    _λa = model.params.lambda_a
    _σ = model.params.sigma

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*_d.^3)/π

    z = ρ̄ /sum(ρ̄)
    m̄ = dot(z,m)
    m̄inv = 1/m̄
    ∑z = sum(z)

    ρS = sum(ρ̄.*m)

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
    
    a₁ = zero(V+T+first(z)+one(eltype(model)))
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
    ρ̄ = sum(ρ̄)
    return ρ̄*adisp
end

function f_assoc(model::SAFTVRMieModel, T, n, n₃, nᵥ)
    HSd = d(model,1e-3,T,onevec(model))
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ))
    nn = assoc_pair_length(model)
    iszero(nn) && return _0

    n₀ = n./HSd
    n₂ = π.*HSd.*n

    nᵥ₂ = -2π.*nᵥ

    ξ = 1 .-nᵥ₂.^2 ./ n₂.^2

    X_ = X(model,T,n,n₃,nᵥ)
    _0 = zero(first(X_.v))

    ns = model.sites.n_sites
    res = _0
    resᵢₐ = _0
    for i ∈ @comps
        ni = ns[i]
        iszero(length(ni)) && continue
        Xᵢ = X_[i]
        resᵢₐ = _0
        for (a,nᵢₐ) ∈ pairs(ni)
            Xᵢₐ = Xᵢ[a]
            nᵢₐ = ni[a]
            resᵢₐ +=  nᵢₐ* (log(Xᵢₐ) - Xᵢₐ/2 + 0.5)
        end
        res += resᵢₐ*n₀[i]*ξ[i]
    end
    return res
end

function Δ(model::SAFTVRMieModel, T, n, n₃, nᵥ, i, j, a, b)
    _d = d(model,1e-3,T,onevec(model))
    _σ = model.params.sigma.values
    m = model.params.segment.values
    ϵ_assoc = model.params.epsilon_assoc.values
    K = model.params.bondvol.values[i,j][a,b]
    iszero(K) && return _0

    ρ̄ = n₃*3*2 ./(_d.^3)/π

    z = ρ̄ /sum(ρ̄)
    m̄ = dot(z,m)
    m̄inv = 1/m̄

    ρS = sum(ρ̄.*m)

    σ3_x = zero(T+first(z)+one(eltype(model)))

    for i ∈ @comps
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        for j ∈ 1:(i-1)
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

export F_res, δFδρ_res