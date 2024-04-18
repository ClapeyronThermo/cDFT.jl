function interfacial_tension(model::EoSModel,p,T,n)
    L = length_scale(model)

    structure = SurfaceTension1DCart((p, T, x),(-10L,10L), 201)

    system = DFTSystem(model, structure)

    converge!(system)

    ρ = system.profiles
    ngrid = system.structure.ngrid

    F = free_energy(system)

    ρl =[ρ[i].boundary_conditions[2] for i in @comps]
    x = ρl/sum(ρl)
    vl = 1/sum(ρl)

    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)

    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(ones(ngrid),ρ[1].mesh_size)
end

interfacial_tension(system::DFTSystem) = surface_tension(system)

export interfacial_tension