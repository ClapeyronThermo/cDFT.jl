"""
    surface_tension(model::EoSModel,T,x)

Calculate the surface tensions of a vapour-liquid interface based on a given `model` and at saturated conditions `T` and `x` (liquid composition). This is a blind calculation that assumes the interface is 1D and cartesian, and that the system does phase split between two bulk phases. If the system is expected to form micelles, other methods should be used.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> surface_tension(model,298.15,[1.0])
```
"""
function surface_tension(model::EoSModel, T,x = [1.0])
    L = length_scale(model)

    (p, vl, vv, y) = bubble_pressure(model, T, x)

    structure = SurfaceTension1DCart((p, T, x),[-10L,10L], 101)

    system = DFTSystem(model, structure)

    converge!(system)

    ρ = system.profiles
    ngrid = system.structure.ngrid

    F = free_energy(system)

    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    chem_pot_term = 0.
    bead_id = 1
    species_id = 1
    for i in 1:length(ρ)
        chem_pot_term += μ[species_id]*∫(ρ[i].density,ρ[i].mesh_size)/system.species[species_id].nbeads
        bead_id += 1
        if bead_id > system.species[species_id].nbeads
            bead_id = 1
            species_id += 1
        end
    end
    return F*k_B*T-chem_pot_term+p*∫(ones(ngrid),ρ[1].mesh_size)
end

function surface_tension(system::DFTSystem)
    model = system.model
    ρ = system.profiles
    ngrid = system.structure.ngrid

    F = free_energy(system)

    (p, T, x) = system.structure.conditions

    ρl =[ρ[i].boundary_conditions[2].value for i in @comps]
    x = ρl/sum(ρl)
    vl = 1/sum(ρl)

    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    chem_pot_term = 0.
    bead_id = 1
    species_id = 1
    for i in 1:length(ρ)
        chem_pot_term += μ[species_id]*∫(ρ[i].density,ρ[i].mesh_size)/system.species[species_id].nbeads
        bead_id += 1
        if bead_id > system.species[species_id].nbeads
            bead_id = 1
            species_id += 1
        end
    end
    return F*k_B*T-chem_pot_term+p*∫(ones(ngrid),ρ[1].mesh_size)
end

export surface_tension