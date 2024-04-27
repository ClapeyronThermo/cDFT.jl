function initialize_profiles(model::EoSModel,structure::InterfacialTension1DCart, species)
    nc = length(model)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, n) = structure.conditions
    n_II = structure.composition_II
    bounds = structure.bounds
    z_interface = sum(bounds)/2

    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    v1 = volume(model,p,T,n)
    v2 = volume(model,p,T,n_II)

    ρ1 = n./v1
    ρ2 = n_II./v2

    k0 = ones(nc)*4
    c0 = ones(nc)*z_interface
    ρ,z = _initial_interfacial_tension_density_profile(model,species,ρ1,ρ2,bounds,z,k0,c0)    

    return ρ
end

function _initial_interfacial_tension_density_profile(model::EoSModel,species,ρ1,ρ2,bounds,z,coef=ones(length(model)),shift=zeros(length(model)))
    L = length_scale(model)

    ρ = DensityProfile[]
    for i in @comps
        nbeads = species[i].nbeads
        for j in 1:nbeads
            boundary_conditions = (FixedBoundary(ρ2[i],-1),FixedBoundary(ρ1[i],1))

            ρ_points = @. tanh_prof(z,ρ1[i],ρ2[i],shift[i],coef[i]/L)

            push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
        end
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

function initialize_profiles(model::EoSModel,structure::InterfacialTension1DSphr, species)
    L = length_scale(model)
    nc = length(model)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, n) = structure.conditions
    n_core = structure.core_composition
    r_interface = structure.r_interface

    r = range(first(bounds),last(bounds),ngrid) |> collect

    v_bulk = volume(model,p,T,n)
    v_core = volume(model,p,T,n_core)

    ρ1 = n./v_bulk
    ρ2 = n_core./v_core

    coeff = ones(nc)*4
    shift = ones(nc)*r_interface

    ρ = DensityProfile[]
    for i in @comps
        nbeads = species[i].nbeads
        for j in 1:nbeads
            boundary_conditions = (FreeBoundary(ρ2[i],-1),FixedBoundary(ρ1[i],1))

            ρ_points = @. tanh_prof(r,ρ1[i],ρ2[i],shift[i],coeff[i]/L)

            push!(ρ,DensityProfile(ρ_points,r,bounds,boundary_conditions))
        end
    end

    return ρ
end