
export initial_interfacial_density_profile, initial_uniform_density_profile

function initial_interfacial_density_profile(model::EoSModel,T,bounds,ngrid::Int64=1001)
    z = range(bounds[1],bounds[2],ngrid)
    z = [z[i] for i in 1:ngrid]


    (Tc,pc,vc) = crit_pure(model)
    (p,vl,vv) = saturation_pressure(model,T)

    ρl = 1/vl
    ρv = 1/vv

    boundary_conditions = [ρv,ρl]

    σ = model.params.sigma.values[1]
    ρ =@. 1/2*(ρl-ρv)*tanh(z/σ*(2.4728-2.3625*T/Tc))+1/2*(ρl+ρv)

    ρ = DensityProfile(ρ,z,bounds,boundary_conditions)
    return ρ, z
end

function initial_uniform_density_profile(model::EoSModel,ρ,bounds,ngrid::Int64=1001)
    z = range(bounds[1],bounds[2],ngrid)
    z = [z[i] for i in 1:ngrid]

    boundary_conditions = [ρ,ρ]

    ρ = ρ*ones(length(z))


    ρ = DensityProfile(ρ,z,bounds,boundary_conditions)
    return ρ, z
end