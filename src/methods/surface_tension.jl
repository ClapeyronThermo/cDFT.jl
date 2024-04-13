function surface_tension(model::EoSModel, T,x = [1.0])
    L = length_scale(model)

    (p, vl, vv, y) = bubble_pressure(model, T, x)

    structure = SurfaceTension1DCart((p, T, x),(-20L,20L), 201)

    system = DFTSystem(model, structure)

    converge!(system)

    ρ = system.profiles
    ngrid = system.structure.ngrid

    F = free_energy(system)

    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(ones(ngrid),ρ[1].mesh_size)
end



function surface_tension(system::DFTSystem)
    model = system.model
    ρ = system.profiles
    ngrid = system.structure.ngrid

    F = free_energy(system)

    (p, T, x) = system.structure.conditions

    ρl =[ρ[i].boundary_conditions[2] for i in @comps]
    vl = 1/sum(ρl)

    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(ones(ngrid),ρ[1].mesh_size)
end

export surface_tension