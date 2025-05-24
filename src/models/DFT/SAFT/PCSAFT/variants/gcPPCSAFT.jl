using Clapeyron: HomogcPCPSAFTModel

function get_species(model::HomogcPCPSAFTModel,structure::DFTStructure)
    return get_species(model.ppcmodel,structure)
end

function f_res(system::DFTSystem, model::HomogcPCPSAFTModel,n)
    return f_res(system,model.ppcmodel,n)
end

function length_scale(model::HomogcPCPSAFTModel)
    return length_scale(model.ppcmodel)
end