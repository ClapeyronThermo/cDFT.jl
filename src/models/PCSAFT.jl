
function f_res(model::PCSAFTModel, T, n, n₃, nᵥ)
    return f_hs(model,T, n, n₃, nᵥ)+
           f_hc(model,T, n, n₃, nᵥ)+
           f_disp(model,T, n, n₃, nᵥ)
end

function F_res(model::PCSAFTModel,ρ,T,z)
    HSd = d(model,[],T,[1.])[1]
    dz = ρ.mesh_size

    lim = 1/2*HSd

    (n, n₃,nᵥ)  = weights_hs(model,ρ,z,lim)

    n₀ = n./HSd

    Φ = f_res.(Ref(model), Ref(T), n, n₃, nᵥ)
    return ∫(Φ,dz) ./ ∫(n₀,dz)
end

function δFδρ_res(model::PCSAFTModel,ρ,T,z)
    HSd = d(model,[],T,[1.])[1]
    lim = 1/2*HSd
    bounds = ρ.bounds.+[-lim,lim]
    mesh_size = ρ.mesh_size

    (n, n₃, nᵥ)  = weights_hs(model,ρ,z,lim)

    z_damp = 0:mesh_size:bounds[2]
    zu = [z_damp[i] for i in 1:length(z_damp)]
    zd = [-z_damp[i] for i in length(z_damp):-1:2]
    z_damp = vcat(zd,zu)

    f(x) = f_res(model,T,x[1],x[2],x[3])
    δf(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(δf,hcat([n n₃ nᵥ]);dims=2)
    ∂f∂n = δfδn[:,1]
    ∂f∂n₃ = δfδn[:,2]
    ∂f∂nᵥ = δfδn[:,3]

    ∂f∂n = DensityProfile(∂f∂n,z_damp,bounds,[∂f∂n[1],∂f∂n[end]])
    ∂f∂n₃ = DensityProfile(∂f∂n₃,z_damp,bounds,[∂f∂n₃[1],∂f∂n₃[end]])
    ∂f∂nᵥ = DensityProfile(∂f∂nᵥ,z_damp,bounds,[∂f∂nᵥ[1],∂f∂nᵥ[end]])

    span = range(-lim,lim,length=101)

    δFδρ_1 = ∫ρdz.(Ref(∂f∂n),z,Ref(span))
    δFδρ_2 = π*∫ρz²dz.(Ref(∂f∂n₃),z,Ref(span))
    δFδρ_3 = -∫ρzdz.(Ref(∂f∂nᵥ),z,Ref(span))

    return δFδρ_1+δFδρ_2+δFδρ_3
end

function f_disp(model::PCSAFTModel, T, n, n₃, nᵥ)
    HSd = d(model,[],T,[1.])[1]

    ϵ = model.params.epsilon.values[1]/T
    σ = model.params.sigma.values[1]
    m = model.params.segment.values[1]

    n₀ = n./HSd

    C₁ = 1+m*(8*n₃-2*n₃^2)/(1-n₃)^4+(1-m)*(20*n₃-27*n₃^2+12*n₃^3-2*n₃^4)/((1-n₃)^2*(2-n₃)^2)
    I₁ = I(model,m,n₃,1)
    I₂ = I(model,m,n₃,2)

    return -π*m*n₀^2*(2*I₁*ϵ+m*C₁^-1*I₂*ϵ^2)*σ^3
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