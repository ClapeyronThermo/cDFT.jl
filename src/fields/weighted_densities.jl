struct WeightedDensity <: DFTField 
    type::Symbol
    width::Vector{Float64}
end

function evaluate_field(system::DFTSystem,field::WeightedDensity)
    ρ = system.profiles
    structure = system.structure
    ngrid = structure.ngrid
    model = system.model
    n = zeros(length(ρ[1].coords),length(ρ))
    width = field.width 
    type = field.type
    size = system.species.size

    if type == :∫ρdz
        integral_method = ∫ρdz
    elseif type == :∫ρzdz
        integral_method = ∫ρzdz
    elseif type == :∫ρz²dz
        integral_method = ∫ρz²dz
    elseif type == :ρ
        for i in @comps
            n[:,i] .= ρ[i].density*N_A
        end
        return n
    else
        error("Invalid type of field")
    end
    
    z = ρ[1].coords

    for i in @comps
        span = range(-width[i],width[i],length=41).*size[i]

        Threads.@threads for j in 1:ngrid
            n[j,i] = integral_method(structure,ρ[i],z[j],span)*N_A
        end
    end
    return n
end

function integrate_field(system::DFTSystem,field::WeightedDensity,profile)
    model = system.model
    structure = system.structure
    nc = length(model)
    ngrid = system.structure.ngrid

    width = field.width 
    type = field.type

    ∫field = zeros(ngrid,nc)
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
        for i in @comps
            ∫field[:,i] .= profile[i].(z)
        end
        return ∫field
    else
        error("Invalid type of field")
    end

    size = system.species.size

    for i in @comps
        span = range(-width[i],width[i],length=41).*size[i]

        Threads.@threads for j in 1:ngrid
            ∫field[j,i] = prefactor*integral_method(structure,profile[i],z[j],span)
        end
    end
    return ∫field
end