"""
    WeightedDensity(type::Symbol,width::Vector{Float64})

Generic `WeightedDensity` type used to calculate the weighted densities of the system. One must specify:
- `type`: The type of weighted density to be calculated. Options are `:∫ρdz` (``n_0``), `:∫ρzdz` (``n_v``), `:∫ρz²dz` (``n_3``), and `:ρ` (unweighted).
- `width`: The width of the weighted density profile.
"""
struct WeightedDensity <: DFTField 
    type::Symbol
    width::Vector{Float64}
end

function evaluate_field(system::DFTSystem,field::WeightedDensity)
    ρ = system.profiles
    structure = system.structure
    ngrid = structure.ngrid
    model = system.model
    species = system.species
    nb = length(ρ)
    ngrid = structure.ngrid
    n = zeros(ngrid,nb)
    width = field.width 
    type = field.type

    if type == :∫ρdz
        integral_method = ∫ρdz
    elseif type == :∫ρzdz
        integral_method = ∫ρzdz
    elseif type == :∫ρz²dz
        integral_method = ∫ρz²dz
    elseif type == :ρ
        for i in 1:nb
            n[:,i] .= ρ[i].density*N_A
        end
        return n
    else
        error("Invalid type of field")
    end
    
    z = ρ[1].coords

    species_id = 1
    bead_id = 1
    for i in 1:nb
        size = species[species_id].size[bead_id]
        span = width[i].*size

        Threads.@threads for j in 1:ngrid
            n[j,i] = integral_method(structure,ρ[i],z[j],span)*N_A
        end

        if bead_id == species[species_id].nbeads
            species_id += 1
            bead_id = 1
        else
            bead_id += 1
        end
    end
    return n
end

function integrate_field(system::DFTSystem,field::WeightedDensity,profile)
    model = system.model
    structure = system.structure
    nb = length(system.profiles)
    species = system.species
    ngrid = system.structure.ngrid

    width = field.width 
    type = field.type

    ∫field = zeros(ngrid,nb)
    z = profile[1].coords

    if type == :∫ρdz
        integral_method = ∫ρdz
        prefactor = 1
    elseif type == :∫ρzdz
        integral_method = ∫ρzdz
        prefactor = -1
    elseif type == :∫ρz²dz
        integral_method = ∫ρz²dz
        prefactor = 1
    elseif type == :ρ
        for i in 1:nb
            ∫field[:,i] .= profile[i].(z)
        end
        return ∫field
    else
        error("Invalid type of field")
    end

    species_id = 1
    bead_id = 1
    for i in 1:nb
        size = species[species_id].size[bead_id]
        span = width[i].*size

        Threads.@threads for j in 1:ngrid
            ∫field[j,i] = prefactor*integral_method(structure,profile[i],z[j],span)
        end

        if bead_id == species[species_id].nbeads
            species_id += 1
            bead_id = 1
        else
            bead_id += 1
        end
    end
    return ∫field
end