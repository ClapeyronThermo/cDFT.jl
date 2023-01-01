function interfacial_tension(model::EoSModel,T)
    σ = model.params.sigma[1]
    (p,vl,vv) = saturation_pressure(model,T)
    ρ,z = initial_interfacial_density_profile(model,T,[-10σ,10σ],101);

    ρ = converge_profile!(model,ρ,T,z)

    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,vl,T,[1.])[1]
    return F*k_B*T-(μ*∫(ρ.(z),ρ.mesh_size)+p*∫(one.(z),ρ.mesh_size))
end

export interfacial_tension