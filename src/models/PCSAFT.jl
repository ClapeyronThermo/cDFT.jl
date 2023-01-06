function F_res(model::PCSAFTModel,ρ,T,z)
    ψ = 1.3862
    HSd = d(model,[],T,ones(length(model)))
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

    f1(x) = f_hs(model,T,x[idx],x[idx.+nc],x[idx.+2*nc])
    Φ_hs = mapslices(f1,hcat([n n₃ nᵥ]);dims=2)

    f2(x) = f_hc(model,T,x[idx],x[idx.+nc],x[idx.+2*nc])
    Φ_hc = mapslices(f2,hcat([ρhc ρ̄hc λ]);dims=2)
    
    f3(x) = f_disp(model,T,x[idx])
    Φ_disp = mapslices(f3,ρ̄;dims=2)
    
    Φ = Φ_hs+Φ_hc+Φ_disp

    ρ_t = mapslices(sum,ρhc;dims=2)
    return ∫(Φ,dz)
end

function δFδρ_res(model::PCSAFTModel,ρ,T,z)
    return δFδρ_hs(model,ρ,T,z)+
           δFδρ_hc(model,ρ,T,z)+
           δFδρ_disp(model,ρ,T,z)
end

function δFδρ_hc(model::PCSAFTModel,ρ,T,z)
    HSd = d(model,[],T,ones(length(model)))
    lim = HSd

    (λ, ρ̄hc,_)  = weights_hs(model,ρ,z,lim)
    ρhc = zeros(length(z),length(ρ))
    for i in @comps
        ρhc[:,i] = ρ[i].density*N_A
    end

    nc = length(model)
    idx = 1:nc
    f(x) = f_hc(model,T,x[idx],x[idx.+nc],x[idx.+2*nc])
    df(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(df,hcat([ρhc ρ̄hc λ]);dims=2)
    ∂f∂ρhc0 = δfδn[:,idx]
    ∂f∂ρ̄hc0 = δfδn[:,idx.+nc]
    ∂f∂λ0 = δfδn[:,idx.+2*nc]

    δFδρ_hc = zeros(length(z),length(model))
    for i in @comps 
        bounds = ρ[i].bounds.+[-lim[i],lim[i]]
        ∂f∂ρhc = DensityProfile(∂f∂ρhc0[:,i],z,bounds,[∂f∂ρhc0[1,i],∂f∂ρhc0[end,i]])
        ∂f∂ρ̄hc = DensityProfile(∂f∂ρ̄hc0[:,i],z,bounds,[∂f∂ρ̄hc0[1,i],∂f∂ρ̄hc0[end,i]])
        ∂f∂λ = DensityProfile(∂f∂λ0[:,i],z,bounds,[∂f∂λ0[1,i],∂f∂λ0[end,i]])
    
        span = range(-lim[i],lim[i],length=101)

        δFδρ_hc_1 = ∫ρdz.(Ref(∂f∂λ),z,Ref(span))
        δFδρ_hc_2 = π*∫ρz²dz.(Ref(∂f∂ρ̄hc),z,Ref(span))
        δFδρ_hc_3 = ∂f∂ρhc.(z)

        δFδρ_hc[:,i] = δFδρ_hc_1+δFδρ_hc_2+δFδρ_hc_3
    end
    return δFδρ_hc
end

function δFδρ_disp(model::PCSAFTModel,ρ,T,z)
    HSd = d(model,[],T,ones(length(model)))
    lim = 1.3862*HSd

    (_, ρ̄,_)  = weights_hs(model,ρ,z,lim)

    nc = length(model)
    idx = 1:nc
    f(x) = f_disp(model,T,x[idx])
    df(x) = ForwardDiff.gradient(f,x)

    δfδn0  = mapslices(df,ρ̄;dims=2)
    ∂f∂n0 = δfδn0[:,idx]

    δFδρ_disp = zeros(length(z),length(model))
    for i in @comps 
        bounds = ρ[i].bounds.+[-lim[i],lim[i]]
        ∂f∂n =  DensityProfile(∂f∂n0[:,i],z,bounds,[∂f∂n0[1,i],∂f∂n0[end,i]])
    
        span = range(-lim[i],lim[i],length=101)

        δFδρ_disp[:,i] = π*∫ρz²dz.(Ref(∂f∂n),z,Ref(span))
    end

    return δFδρ_disp
end

function f_hc(model::PCSAFTModel, T, ρhc, ρ̄hc, λ)
    HSd = d(model,[],T,ones(length(model)))
    m = model.params.segment.values

    ζ₃ = sum(1/8*m.*ρ̄hc)
    ζ₂ = sum(1/8*m.*ρ̄hc./HSd)
    λ = λ./(2*HSd)
    
    yᵈᵈ = @. 1/(1-ζ₃)+1.5*HSd*ζ₂/(1-ζ₃)^2+0.5*HSd^2*ζ₂^2/(1-ζ₃)^3
    f = @. -ρhc*(m-1)*log(yᵈᵈ*λ/ρhc)
    return sum(f)
end

function f_disp(model::PCSAFTModel, T, ρ̄)
    ψ = 1.3862
    HSd = d(model,[],T,ones(length(model)))
    σ = model.params.sigma.values
    m = model.params.segment.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π

    x = ρ̄/sum(ρ̄)
    m̄ = sum(x.*m)

    η = π/6*sum(ρ̄.*m.*HSd.^3)

    C₁ = 1+m̄*(8*η-2*η^2)/(1-η)^4+(1-m̄)*(20*η-27*η^2+12*η^3-2*η^4)/((1-η)^2*(2-η)^2)
    I₁ = I(model,m̄,η,1)
    I₂ = I(model,m̄,η,2)

    m2ϵσ3₁,m2ϵσ3₂ =  m2ϵσ3(model, T, x)
    ρ̄ = sum(ρ̄)

    return -2*π*ρ̄^2*I₁*m2ϵσ3₁-π*ρ̄^2*m̄*C₁^-1*I₂*m2ϵσ3₂
end

function m2ϵσ3(model::PCSAFTModel, T, x)
    m = model.params.segment.values
    σ = model.params.sigma.values
    ϵ = model.params.epsilon.values
    m2ϵσ3₂ = zero(T+first(x))
    m2ϵσ3₁ = m2ϵσ3₂
    @inbounds for i ∈ @comps
        for j ∈ @comps
            constant = x[i]*x[j]*m[i]*m[j] * σ[i,j]^3
            exp1 = (ϵ[i,j]/T)
            exp2 = exp1*exp1
            m2ϵσ3₁ += constant*exp1
            m2ϵσ3₂ += constant*exp2
        end
    end
    return m2ϵσ3₁,m2ϵσ3₂
    #return ∑(z[i]*z[j]*m[i]*m[j] * (ϵ[i,j]*(1)/T)^n * σ[i,j]^3 for i ∈ @comps, j ∈ @comps)/(sum(z)^2)
end

function I(model::PCSAFTModel,m̄,n₃,n)
    if n == 1
        corr = Clapeyron.PCSAFTconsts.corr1
    elseif n == 2
        corr = Clapeyron.PCSAFTconsts.corr2
    end
    res = zero(n₃)
    @inbounds for i ∈ 1:7
        ii = i-1 
        corr1,corr2,corr3 = corr[i]
        ki = corr1 + (m̄-1)/m̄*corr2 + (m̄-1)/m̄*(m̄-2)/m̄*corr3
        res += ki*n₃^ii
    end
    return res
end

# function f_assoc(model::PCSAFTModel, T, n, n₃, nᵥ)
#     _0 = zero(V+T+first(z))
#     nn = Clapeyron.assoc_pair_length(model)
#     iszero(nn) && return _0
#     X_ = X()

#     return -n₀*log(1-n₃)+(n₁*n₂-nᵥ₂*nᵥ₁)/(1-n₃)+(n₂^3/3-n₂*nᵥ₂*nᵥ₂)*(log(1-n₃)/(12*π*n₃^2)+1/(12*π*n₃*(1-n₃)^2))
# end

# function Δ(model::PCSAFTModel, T, n₀, n₂, n₃, nᵥ₂, a, b)
#     ϵ_assoc = model.params.epsilon_assoc.values
#     κ = model.params.bondvol.values
#     κijab = κ[1,1][a,b] 
#     iszero(κijab) && return _0

#     σ = model.params.sigma.values[1]
#     m = model.params.segment[1]
#     HSd = d(model,[],T,[1.])[1]

#     ξ = 1-nᵥ₂^2/n₂^2
#     g_hs = 1/(1-n₃)+HSd*ξ*n₂/(2*(1-n₃)^2)+HSd^2*n₂^2*ξ/(18*(1-n₃)^3)
#     return g_hs*σ*(exp(ϵ_assoc[i,j][a,b]/T)-1)*κijab
# end

export F_res, δFδρ_res