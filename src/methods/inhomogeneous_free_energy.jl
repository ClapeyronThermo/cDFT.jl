function eval_interfacial_tension(model::EoSModel,p,T,ρ,z)
    ρ_bound = [ρ[i].boundary_conditions[1] for i in @comps]

    v = 1/sum(ρ_bound)
    x = ρ_bound/sum(ρ_bound)
    
    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,v,T,x)

    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(one.(z),ρ[1].mesh_size)
end