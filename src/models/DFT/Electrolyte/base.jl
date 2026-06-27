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

# ── Enzyme/KA kernel path for ElectrolyteDFTSystem ──────────────────────────

function preallocate_params(system::ElectrolyteDFTSystem)
    model   = system.model
    n_model = model.neutralmodel
    nd      = dimension(system)

    # Drop the last field (ion) to build a proxy neutral DFTSystem
    n_fields   = Base.front(system.fields)
    NF_neutral = compute_field_len(n_fields, nd)
    n_sys = DFTSystem(n_model, system.species, system.structure,
                      n_fields, nothing, system.propagator,
                      system.options, Val{NF_neutral}())
    neutral_params, nc = preallocate_params(n_sys)

    # Ion model params — dispatch to specific ion model implementation
    ion_params = preallocate_params(system, model.ionmodel)

    return merge(neutral_params, ion_params), nc
end

@inline function f_res(out, n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: ElectrolyteModel}
    MN = fieldtype(M, :neutralmodel)
    MI = fieldtype(M, :ionmodel)
    f_res(out, n, params, T, kk, Val(NC), Val(ND), MN)
    f_res(out, n, params, T, kk, Val(NC), Val(ND), MI)
    return nothing
end