function interfacial_tension(model::EoSModel,p,T,n;surfactant=nothing,K0=nothing)
    σ = maximum(model.params.sigma.values)

    ρ,z = initial_interfacial_tension_density_profile(model,p,T,n,[-10σ,10σ],201;surfactant=surfactant,K0=K0)

    converge_profile!(model,ρ,T,z;damping=0.01)

    return eval_interfacial_tension(model,p,T,ρ,z)
end

function initial_interfacial_tension_density_profile(model::EoSModel,p,T,n,bounds,ngrid::Int64=101;surfactant=nothing,K0=nothing)
    nc = length(model)    
    components = model.components
    icomponents = 1:length(model)
    isurf = icomponents[components.==surfactant]

    z = range(first(bounds),last(bounds),ngrid) |> collect

    (x,N,G) = tp_flash(model,p,T,n,RRTPFlash(equilibrium=:lle))
    println(x)

    v1 = volume(model,p,T,x[1,:])
    v2 = volume(model,p,T,x[2,:])

    ρ1 = x[1,:]./v1
    ρ2 = x[2,:]./v2

    f(x) = obj_initial_profile(model,p,T,ρ1,ρ2,bounds,z,x[1:nc],x[nc+1:2*nc];surfactant=surfactant)
    k0 = ones(nc)
    c0 = zeros(nc)
    c0[isurf] .= 10
    z0 = vcat(k0,c0)
    res =  optimize(f,z0,Optim.NelderMead(),Optim.Options())
    x_sol = Optim.minimizer(res)
    k = x_sol[1:nc]
    c = x_sol[nc+1:2*nc]

    ρ,z = _initial_interfacial_tension_density_profile(model,ρ1,ρ2,bounds,z,k,c)
    return ρ, z
end


# ρ(z) = 1/2(ρ1-ρ2)*tanh(a*z/σ+b)+1/2*(ρ1+ρ2)+c^2*exp(-(z/\sigma))
function _initial_interfacial_tension_density_profile(model::EoSModel,ρ1,ρ2,bounds,z,coef=ones(length(model)),shift=zeros(length(model));surfactant=nothing)
    ρ = DensityProfile[]
    components = model.components
    icomponents = 1:length(model)
    isurf = icomponents[components.==surfactant]
    for i in @comps
        boundary_conditions = [ρ2[i],ρ1[i]]
        σ = model.params.sigma[i]
        if i in isurf
            ρ_points =@. 1/2*(ρ1[i]-ρ2[i])*tanh(z/σ)+1/2*(ρ1[i]+ρ2[i])+maximum([ρ1[i],ρ2[i]])[1]*shift[i]^2*exp(-(z/σ)^2/coef[i]^2)
        else
            ρ_points =@. 1/2*(ρ1[i]-ρ2[i])*tanh(z/σ*coef[i]+shift[i])+1/2*(ρ1[i]+ρ2[i])
        end

        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

function obj_initial_profile(model,p,T,ρ1,ρ2,bounds,z,k,c;surfactant=nothing)
    ρ, z = _initial_interfacial_tension_density_profile(model,ρ1,ρ2,bounds,z,k,c;surfactant=surfactant)
    return eval_interfacial_tension(model,p,T,ρ,z)
end

export interfacial_tension