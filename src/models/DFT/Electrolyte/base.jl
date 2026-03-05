import Clapeyron: ElectrolyteModel

include("electrostatic_potential.jl")
include("DH.jl")


function ElectrolyteDFTSystem(model::ElectrolyteModel, structure::DFTStructure, external_field::ExternalFieldModel, options::DFTOptions = DFTOptions())
    species = get_species(model.neutralmodel, structure)
    ion_species = get_species(model.ionmodel, model.neutralmodel, model.charge, structure)

    (p,T) = structure.conditions
    ρbulk = structure.ρbulk
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T

    species.chempot_res .= μres


    fields = get_fields(model.neutralmodel, species, structure, options.device)
    fields_ion = get_fields(model.ionmodel, ion_species, structure, options.device)
    append!(fields,fields_ion)

    typed_fields = tuple(fields...)

    propagator = get_propagator(model.neutralmodel, species, structure)
    

    external_field = [external_field, ElectrostaticPotential(model, structure, options.device)]

    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return ElectrolyteDFTSystem(model, species, ion_species, structure, typed_fields, external_field, propagator, options,chunksize)
end

function ElectrolyteDFTSystem(model::ElectrolyteModel, structure::DFTStructure, options::DFTOptions = DFTOptions())
    device = options.device
    species = get_species(model.neutralmodel, structure)
    ion_species = get_species(model.ionmodel, model.neutralmodel, model.charge, structure)

    (p,T) = structure.conditions
    ρbulk = structure.ρbulk
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T

    species.chempot_res .= μres


    fields = get_fields(model.neutralmodel, species, structure, device)
    fields_ion = get_fields(model.ionmodel, ion_species, structure, device)
    append!(fields,fields_ion)

    typed_fields = tuple(fields...)

    propagator = get_propagator(model.neutralmodel, species, structure)

    external_field = [ElectrostaticPotential(model, structure, device)]

    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return ElectrolyteDFTSystem(model, species, ion_species, structure, typed_fields, external_field, propagator, options,chunksize)
end

function f_res(system::ElectrolyteDFTSystem, model::ElectrolyteModel,n)
    n_neutral = @view(n[1:end-1,:])
    n_ion = @view(n[end,:])
    return f_res(system,model.neutralmodel,n_neutral) + f_res(system,model.ionmodel,n_ion)
end

length_scale(model::ElectrolyteModel) = length_scale(model.neutralmodel)