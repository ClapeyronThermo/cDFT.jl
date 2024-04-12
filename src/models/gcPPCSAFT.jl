using Clapeyron: gcPCPSAFTModel

function F_res(model::gcPCPSAFTModel,ρ,T,z)
    return F_res(model.ppcmodel,ρ,T,z)
end

function δFδρ_res(model::gcPCPSAFTModel,ρ,T,z)
    return δFδρ_res(model.ppcmodel,ρ,T,z)
end

function length_scale(model::gcPCPSAFTModel)
    return length_scale(model.ppcmodel)
end