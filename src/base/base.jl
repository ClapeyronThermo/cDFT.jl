abstract type DFTProfile end
abstract type DFTField end

include("devices.jl")
include("structure.jl")

struct DFTSystem
    model::EoSModel
    structure::DFTStructure
    profiles::Vector{DFTProfile}
    fields::Vector{DFTField}
    options::DFTOptions
end

struct DFTOptions
    device::Device
    solver::Solvers.AbstractFixPoint
end

function DFTSystem(model::EoSModel,structure::DFTStructure,profile::Vector{Profile},fields::Vector{Function},options::DFTOptions)
    return DFTSystem(model,structure,profile,fields,options)
end

function DFTSystem(model::PCSAFTModel, structure::DFTStructure, options::DFTOptions = CPU())
    profiles = initialize_profiles(model,structure)
    fields = get_fields(model)
    return DFTSystem(model, structure, profiles, fields, options)
end
