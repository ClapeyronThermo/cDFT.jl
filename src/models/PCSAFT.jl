abstract type PCSAFTFunctionalModel <: SAFTFunctionalModel end

struct PCSAFTFunctional{T <: EoSModel} <: PCSAFTFunctionalModel
    eosmodel::T
    coords::Vector{Float64}
    density::Vector{Float64}
    coords_full::Vector{Float64}
    density_full::Vector{Float64}
    bounds::Float64
end

export PCSAFTFunctional

function PCSAFTFunctional(eosmodel::EoSModel,width::Float64;dz::Float64=0.01)
    σ = maximum(eosmodel.params.sigma.values)
    N = Int(2*(width+2)/dz)+1
    coords_full = range(-width-2,width+2,length=N)
    coords_full = [coords_full[i] for i in 1:N]
    coords = coords_full[-width .<=coords_full .<=width]
    return PCSAFTFunctional(eosmodel,coords,zeros(length(coords)),coords_full,zeros(N),width)
end

function f_res(model::PCSAFTFunctionalModel, T, n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)
    return f_hs(model,T, n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)+
           f_hc(model,T, n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)+
           f_disp(model,T, n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)
end

function F_res(model::PCSAFTFunctionalModel,T)
    z = model.coords
    HSd = d(model.eosmodel,[],T,[1.])[1]
    dz = (z[2]-z[1])*HSd
    
    (n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)  = weights_hs(model,T,z)

    Φ = @. f_res(Ref(model), Ref(T), n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)
    return ∫(Φ,dz) ./ ∫(n₀,dz)
end

function δFδρ_res(model::PCSAFTFunctionalModel,T)
    HSd = d(model.eosmodel,[],T,[1.])[1]

    z = model.coords
    z_full = model.coords_full
    
    idx1 = @. (z[1]-1<=z_full && z_full<=z[end]+1)

    (n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)  = weights_hs(model,T,z_full[idx1])

    f(x) = f_res(model,T,x[1],x[2],x[3],x[4],x[5],x[6])
    df(x) = ForwardDiff.gradient(f,x)

    δfδn  = mapslices(df,hcat([n₀ n₁ n₂ n₃ nᵥ₁ nᵥ₂]);dims=2)
    ∂f∂n₀ = δfδn[:,1]
    ∂f∂n₁ = δfδn[:,2]
    ∂f∂n₂ = δfδn[:,3]
    ∂f∂n₃ = δfδn[:,4]
    ∂f∂nᵥ₁ = δfδn[:,5]
    ∂f∂nᵥ₂ = δfδn[:,6]
        
    δFδρ_1 = ∫fdz.(Ref(1/HSd*∂f∂n₀+1/2*∂f∂n₁+π*HSd*∂f∂n₂),Ref(z_full[idx1]),z,1/2)*HSd
    δFδρ_2 = ∫fz²dz.(Ref(π*∂f∂n₃),Ref(z_full[idx1]),z,1/2)*HSd^3
    δFδρ_3 = ∫fzdz.(Ref(1/HSd*∂f∂nᵥ₁+2π*∂f∂nᵥ₂),Ref(z_full[idx1]),z,1/2)*HSd^2

    return δFδρ_1+δFδρ_2+δFδρ_3
end

function f_disp(model::PCSAFTFunctionalModel, T, n₀, n₁, n₂, n₃, nᵥ₁, nᵥ₂)
    ϵ = model.eosmodel.params.epsilon.values[1]/T
    σ = model.eosmodel.params.sigma.values[1]
    m = model.eosmodel.params.segment.values[1]

    C₁ = 1+m*(8*n₃-2*n₃^2)/(1-n₃)^4+(1-m)*(20*n₃-27*n₃^2+12*n₃^3-2*n₃^4)/((1-n₃)^2*(2-n₃)^2)
    I₁ = I(model,m,n₃,1)
    I₂ = I(model,m,n₃,2)

    return -π*m*n₀^2*(2*I₁*ϵ+m*C₁^-1*I₂*ϵ^2)*σ^3
end

function I(model::PCSAFTFunctionalModel,m̄,n₃,n)
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