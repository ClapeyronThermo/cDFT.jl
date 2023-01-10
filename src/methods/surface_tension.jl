function surface_tension(model::EoSModel,T)
    σ = model.params.sigma[1]
    (Tc,pc,vc) = crit_pure(model)
    (p,vl,vv) = saturation_pressure(model,T)
    # if T<0.75Tc
        ρ,z = initial_interfacial_density_profile(model,T,[-10σ,10σ],101)
    # else
    #     ρ,z = initial_interfacial_density_profile(model,T,[-20σ,20σ],201)
    # end

    ρ = converge_profile!(model,ρ,T,z)

    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,vl,T,[1.])[1]
    return F*k_B*T-μ*∫(ρ[1].density,ρ[1].mesh_size)+p*∫(one.(z),ρ[1].mesh_size)
end

function surface_tension(model::EoSModel,T,x)
    σ = model.params.sigma[1]
    (p,vl,vv) = bubble_pressure(model,T,x)
    # if T<0.75Tc
        ρ,z = initial_interfacial_density_profile(model,T,x,[-10σ,10σ],101)
    # else
    #     ρ,z = initial_interfacial_density_profile(model,T,[-20σ,20σ],201)
    # end

    ρ = converge_profile!(model,ρ,T,z)

    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(one.(z),ρ[1].mesh_size)
end

export surface_tension