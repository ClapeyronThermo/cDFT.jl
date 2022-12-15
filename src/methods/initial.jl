export initial_interfacial_density_profile!, initial_uniform_density_profile!

function initial_interfacial_density_profile!(model::FunctionalModel,T)
    z = model.coords_full
    σ = model.eosmodel.params.sigma.values[1]
    (Tc,pc,vc) = crit_pure(model.eosmodel)
    (p,vl,vv) = saturation_pressure(model.eosmodel,T)

    ρl = 1/vl
    ρv = 1/vv

    ρ =@. 1/2*(ρl-ρv)*tanh(z/σ*(2.4728-2.3625*T/Tc))+1/2*(ρl+ρv)

    model.density_full .= ρ
    model.density .= ρ[-model.bounds .<=z .<=model.bounds]
end

function initial_uniform_density_profile!(model::FunctionalModel,T,ρ)
    z = model.coords_full

    ρ = ρ*ones(length(z))

    model.density_full .= ρ
    model.density .= ρ[-model.bounds .<=z .<=model.bounds]
end