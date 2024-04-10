using Clapeyron: PCSAFTModel

function F_res(model::PCSAFTModel,ρ,T,z)
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
)+f_assoc(model,T,@view(x[idx]),@view(x[idx.+nc]),@view(x[idx.+2*nc]),HSd)

    Φ_hs_assoc = mapslices(f1,[n n₃ nᵥ];dims=2)

    f2(x) = f_hc(model,T,@view(x[idx]),@view(x[idx.+nc]),@view(x[idx.+2*nc])
)
    Φ_hc = mapslices(f2,[ρhc ρ̄hc λ];dims=2)
    
    f3(x) = f_disp(model,T,@view(x[idx]))
    Φ_disp = mapslices(f3,ρ̄;dims=2)
    
    Φ = Φ_hc+Φ_disp+Φ_hs_assoc

    return ∫(Φ,dz)
end

function δFδρ_res(model::PCSAFTModel,ρ,T,z)
    return δFδρ_hs(model,ρ,T,z)+
           δFδρ_hc(model,ρ,T,z)+
           δFδρ_disp(model,ρ,T,z)+
           δFδρ_assoc(model,ρ,T,z)
end

function δFδρ_hc(model::PCSAFTModel,ρ,T,z)
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
    
        span = range(-lim[i],lim[i],length=length(z))

        δFδρ_hc_1 = ∫ρdz.(Ref(∂f∂λ),z,Ref(span))
        δFδρ_hc_2 = π*∫ρz²dz.(Ref(∂f∂ρ̄hc),z,Ref(span))
        δFδρ_hc_3 = ∂f∂ρhc.(z)

        δFδρ_hc[:,i] = δFδρ_hc_1+δFδρ_hc_2+δFδρ_hc_3
    end
    return δFδρ_hc
end

function δFδρ_disp(model::PCSAFTModel,ρ,T,z)
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
        span = range(-lim[i],lim[i],length=length(z))
        δFδρ_disp[:,i] = π*∫ρz²dz.(Ref(∂f∂n),z,Ref(span))
    end

    return δFδρ_disp
end

function δFδρ_assoc(model::SAFTModel,ρ,T,z)
    HSd = d(model,1e-3,T,onevec(model))
    lim = 1/2*HSd

    (n, n₃, nᵥ)  = weights_hs(model,ρ,z,lim)

    (∂f∂n0, ∂f∂n₃0, ∂f∂nᵥ0) = δfδρ_hs(model, T, n, n₃, nᵥ)

    nc = length(model)
    idx = 1:nc
    f(x) = f_assoc(model,T,@view(x[idx]),@view(x[idx.+nc]),@view(x[idx.+2*nc]))
    df(x) = ForwardDiff.gradient(f,x)

    δfδn0  = mapslices(df,[n n₃ nᵥ];dims=2)
    ∂f∂n0 = δfδn0[:,idx]
    ∂f∂n₃0 = δfδn0[:,idx.+nc]
    ∂f∂nᵥ0 = δfδn0[:,idx.+2*nc]

    δFδρ_assoc = zeros(length(z),length(model))
    for i in @comps 
        bounds = ρ[i].bounds.+(-lim[i],lim[i])
        ∂f∂n = DensityProfile(∂f∂n0[:,i],z,bounds,[∂f∂n0[1,i],∂f∂n0[end,i]])
        ∂f∂n₃ = DensityProfile(∂f∂n₃0[:,i],z,bounds,[∂f∂n₃0[1,i],∂f∂n₃0[end,i]])
        ∂f∂nᵥ = DensityProfile(∂f∂nᵥ0[:,i],z,bounds,[∂f∂nᵥ0[1,i],∂f∂nᵥ0[end,i]])
        span = range(-lim[i],lim[i],length=length(z))
        for k in eachindex(z)
            zk = z[k]
            δFδρ_assoc_1 = ∫ρdz(∂f∂n,zk,span)
            δFδρ_assoc_2 = π*∫ρz²dz(∂f∂n₃,zk,span)
            δFδρ_assoc_3 = -∫ρzdz(∂f∂nᵥ,zk,span)
            δFδρ_assoc[k,i] = δFδρ_assoc_1+δFδρ_assoc_2+δFδρ_assoc_3
        end
    end
    return δFδρ_assoc
end

function f_hc(model::PCSAFTModel, T, ρhc, ρ̄hc, _λ)
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

function f_disp(model::PCSAFTModel, T, ρ̄)
    ψ = 1.3862
    HSd = d(model,1e-3,T,onevec(model))
    σ = model.params.sigma.values
    m = model.params.segment.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
    ∑ρ̄ = sum(ρ̄)
    x = ρ̄ /∑ρ̄
    m̄ = dot(x,m)
    η = zero(first(m) + ∑ρ̄ + first(HSd))
    for i in 1:length(m)
        η += m[i]*ρ̄[i]*HSd[i]^3
    end
    η = π/6*η
    C₁ = 1+m̄*(8*η-2*η^2)/(1-η)^4+(1-m̄)*(20*η-27*η^2+12*η^3-2*η^4)/((1-η)^2*(2-η)^2)
    I₁ = I(model,m̄,η,1)
    I₂ = I(model,m̄,η,2)

    m2ϵσ3₁,m2ϵσ3₂ =  Clapeyron.m2ϵσ3(model,zero(T), T, x)
    
    return -2*π*∑ρ̄^2*I₁*m2ϵσ3₁-π*∑ρ̄^2*m̄*C₁^-1*I₂*m2ϵσ3₂
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

function Δ(model::PCSAFTModel, T, n, n₃, nᵥ, i, j, a, b)
    ϵ_assoc = model.params.epsilon_assoc.values
    κ = model.params.bondvol.values
    κijab = κ[i,j][a,b] 
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(κijab))
    iszero(κijab) && return _0

    σ = model.params.sigma.values[i,j]
    m = model.params.segment.values
    HSd = d(model,1e-3,T,onevec(model))
    dij = (HSd[i]*HSd[j])/(HSd[i]+HSd[j])
    
    n₂, nᵥ₂, n₃₃ = _0,_0,_0
    for i in 1:length(n)
        nᵢ,mᵢ,nᵥᵢ,HSdᵢ = n[i],m[i],nᵥ[i],HSd[i]
        n₂ += π*HSdᵢ*nᵢ*mᵢ
        nᵥ₂ += -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[i]*mᵢ
    end
    #n₂ = sum(π.*HSd.*n.*m)
    #nᵥ₂ = sum(-2π.*nᵥ.*m)
    #n₃  = sum(n₃.*m)

    ξ = 1-nᵥ₂^2/n₂^2
    g_hs = 1/(1-n₃₃)+dij*ξ*n₂/(2*(1-n₃₃)^2)+dij^2*n₂^2*ξ/(18*(1-n₃₃)^3)
    return g_hs*σ^3*expm1(ϵ_assoc[i,j][a,b]/T)*κijab
end

function Δ(model::EoSModel, T, n, n₃, nᵥ)
    Δout = assoc_similar(model,typeof(T+first(n₃)+first(n)+first(nᵥ)))
    Δout.values .= false
    for (idx,(i,j),(a,b)) in indices(Δout)
        Δout[idx] = Δ(model,T,n, n₃, nᵥ,i,j,a,b)
    end
    return Δout
end

export F_res, δFδρ_res

function length_scale(model::SAFTModel)
    return maximum(model.params.sigma.values)
end