function interfacial_tension(model::EoSModel,p,T,n)
    σ = maximum(model.params.sigma.values)

    ρ,z = initial_interfacial_tension_density_profile(model,p,T,n,[-20σ,20σ],201;coef=[2.,3.5])

    ρ = converge_profile!(model,ρ,T,z;damping=0.001)

    ρ_bound = [ρ[i].boundary_conditions[1] for i in @comps]

    v = 1/sum(ρ_bound)
    x = ρ_bound/sum(ρ_bound)

    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,v,T,x)
    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(one.(z),ρ[1].mesh_size)
end

export interfacial_tension