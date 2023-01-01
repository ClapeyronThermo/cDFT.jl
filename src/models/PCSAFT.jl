function F_res(model::PCSAFTModel,ρ,T,z)
    ψ = 1.5357
    HSd = d(model,[],T,[1.])[1]
    dz = ρ.mesh_size
    bounds = ρ.bounds

    (n, n₃,nᵥ)  = weights_hs(model,ρ,z,1/2*HSd)
    (λ, ρ̄hc,_)    = weights_hs(model,ρ,z,HSd)
    (_, ρ̄,_)    = weights_hs(model,ρ,z,ψ*HSd)

    ρhc = ρ.(z)*N_A
    
    Φ = f_hs.(Ref(model), Ref(T), n, n₃, nᵥ)+
        f_hc.(Ref(model), Ref(T), ρhc, ρ̄hc, λ)+
        f_disp.(Ref(model), Ref(T), ρ̄)
    return ∫(Φ,dz)
end

function δFδρ_res(model::PCSAFTModel,ρ,T,z)
    return δFδρ_hs(model,ρ,T,z)+
           δFδρ_hc(model,ρ,T,z)+
           δFδρ_disp(model,ρ,T,z)
end

function δFδρ_hc(model::PCSAFTModel,ρ,T,z)
    HSd = d(model,[],T,[1.])[1]
    lim = HSd
    bounds = ρ.bounds.+[-lim,lim]
    mesh_size = ρ.mesh_size

    z_damp = 0:mesh_size:bounds[2]
    zu = [z_damp[i] for i in 1:length(z_damp)]
    zd = [-z_damp[i] for i in length(z_damp):-1:2]
    z_damp = vcat(zd,zu)

    (λ, ρ̄hc,_)  = weights_hs(model,ρ,z,lim)
    ρhc = ρ.(z)*N_A

    f(x) = f_hc(model,T,x[1],x[2],x[3])
    δf(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(δf,hcat([ρhc ρ̄hc λ]);dims=2)
    ∂f∂ρhc = δfδn[:,1]
    ∂f∂ρ̄hc = δfδn[:,2]
    ∂f∂λ = δfδn[:,3]

    ∂f∂ρhc = DensityProfile(∂f∂ρhc,z,bounds,[∂f∂ρhc[1],∂f∂ρhc[end]])
    ∂f∂ρ̄hc = DensityProfile(∂f∂ρ̄hc,z,bounds,[∂f∂ρ̄hc[1],∂f∂ρ̄hc[end]])
    ∂f∂λ = DensityProfile(∂f∂λ,z,bounds,[∂f∂λ[1],∂f∂λ[end]])

    span = range(-lim,lim,length=101)

    δFδρ_1 = ∫ρdz.(Ref(∂f∂λ),z,Ref(span))
    δFδρ_2 = π*∫ρz²dz.(Ref(∂f∂ρ̄hc),z,Ref(span))
    δFδρ_3 = ∂f∂ρhc.(z)
    return δFδρ_1+δFδρ_2+δFδρ_3
end

function δFδρ_disp(model::PCSAFTModel,ρ,T,z)
    HSd = d(model,[],T,[1.])[1]
    lim = 1.5357*HSd
    bounds = ρ.bounds.+[-lim,lim]
    mesh_size = ρ.mesh_size

    z_damp = 0:mesh_size:bounds[2]
    zu = [z_damp[i] for i in 1:length(z_damp)]
    zd = [-z_damp[i] for i in length(z_damp):-1:2]
    z_damp = vcat(zd,zu)

    (_, ρ̄,_)  = weights_hs(model,ρ,z,lim)

    f(x) = f_disp(model,T,x[1])
    δf(x) = ForwardDiff.derivative(f,x)

    δfδn  = δf.(ρ̄)
    ∂f∂ρ̄ = δfδn[:,1]

    ∂f∂ρ̄ = DensityProfile(∂f∂ρ̄,z,bounds,[∂f∂ρ̄[1],∂f∂ρ̄[end]])
    
    span = range(-lim,lim,length=101)

    δFδρ = π*∫ρz²dz.(Ref(∂f∂ρ̄),z,Ref(span))
    return δFδρ
end

function f_hc(model::PCSAFTModel, T, ρhc, ρ̄hc, λ)
    HSd = d(model,[],T,[1.])[1]
    m = model.params.segment.values[1]

    ζ₃ = 1/8*m*ρ̄hc
    λ = λ/(2*HSd)
    
    yᵈᵈ = @. 1/(1-ζ₃)+1.5*ζ₃/(1-ζ₃)^2+0.5*ζ₃^2/(1-ζ₃)^3
    return @. -ρhc*(m-1)*log(yᵈᵈ*λ/ρhc)
end

function f_disp(model::PCSAFTModel, T, ρ̄)
    ψ = 1.5357

    HSd = d(model,[],T,[1.])[1]
    ρ̄ = ρ̄*3/(4*ψ^3*HSd^3)/π

    ϵ = model.params.epsilon.values[1]/T
    σ = model.params.sigma.values[1]
    m = model.params.segment.values[1]

    η = π/6*ρ̄*m*HSd^3

    C₁ = 1+m*(8*η-2*η^2)/(1-η)^4+(1-m)*(20*η-27*η^2+12*η^3-2*η^4)/((1-η)^2*(2-η)^2)
    I₁ = I(model,m,η,1)
    I₂ = I(model,m,η,2)

    return -π*ρ̄^2*m^2*(2*I₁*ϵ+m*C₁^-1*I₂*ϵ^2)*σ^3
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

export F_res, δFδρ_res