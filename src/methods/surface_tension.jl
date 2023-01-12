function surface_tension(model::EoSModel,T,x = SA[1.0])
    σ = maximum(model.params.sigma.values)
    (p,vl,vv,y) = bubble_pressure(model,T,x) #on single component, returns just the saturation pressure.
    # if T<0.75Tc
        ρ,z = initial_surface_tension_density_profile(model,T,x,[-10σ,10σ],101)
    # else
    #     ρ,z = initial_interfacial_density_profile(model,T,[-20σ,20σ],201)
    # end

    ρ = converge_profile!(model,ρ,T,z)
    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(one.(z),ρ[1].mesh_size)
end

export surface_tension