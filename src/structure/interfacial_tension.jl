function initialize_profiles(model::EoSModel,structure::InterfacialTension1DCart)
    nc = length(model)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, n) = structure.conditions

    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    (x,N,G) = tp_flash(model,p,T,n,RRTPFlash(equilibrium=:lle))

    v1 = volume(model,p,T,x[1,:])
    v2 = volume(model,p,T,x[2,:])

    ρ1 = x[1,:]./v1
    ρ2 = x[2,:]./v2

    k0 = ones(nc)*4
    c0 = zeros(nc)
    ρ,z = _initial_interfacial_tension_density_profile(model,ρ1,ρ2,bounds,z,k0,c0)    

    return ρ
end

function _initial_interfacial_tension_density_profile(model::EoSModel,ρ1,ρ2,bounds,z,coef=ones(length(model)),shift=zeros(length(model)))
    L = length_scale(model)

    ρ = DensityProfile[]
    for i in @comps
        boundary_conditions = (FixedBoundary(ρ2[i],-1),FixedBoundary(ρ1[i],1))

        ρ_points =@. 1/2*(ρ1[i]-ρ2[i])*tanh(z/L*coef[i]+shift[i])+1/2*(ρ1[i]+ρ2[i])

        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

function eval_interfacial_tension(model::EoSModel,p,T,ρ,z)
    ρ_bound = [ρ[i].boundary_conditions[1] for i in @comps]

    v = 1/sum(ρ_bound)
    x = ρ_bound/sum(ρ_bound)
    
    F = free_energy(system)
    μ = Clapeyron.VT_chemical_potential(model,v,T,x)

    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(one.(z),ρ[1].mesh_size)
end

function obj_initial_profile(model,p,T,ρ1,ρ2,bounds,z,k,c)
    ρ, z = _initial_interfacial_tension_density_profile(model,ρ1,ρ2,bounds,z,k,c)
    system = DFTSystem(model, 
                       structure,
                       )
    return eval_interfacial_tension(model,p,T,ρ,z)
end