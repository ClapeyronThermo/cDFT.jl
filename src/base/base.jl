abstract type DFTProfile end
abstract type DFTField end
abstract type DFTSpecies end

include("devices.jl")
include("structure.jl")

struct DFTSystem
    model::EoSModel
    species::DFTSpecies
    structure::DFTStructure
    profiles::Vector{DFTProfile}
    fields::Vector{DFTField}
    options::DFTOptions
end

function DFTSystem(model::EoSModel,structure::DFTStructure,profile::Vector{DFTProfile},fields::Vector{DFTField},options::DFTOptions)
    return DFTSystem(model,species,structure,profile,fields,options)
end

function DFTSystem(model::EoSModel, structure::DFTStructure, options::DFTOptions = DFTOptions())
    profiles = initialize_profiles(model,structure)
    fields = get_fields(model)
    species = get_species(model, structure)
    return DFTSystem(model, species, structure, profiles, fields, options)
end

export DFTSystem