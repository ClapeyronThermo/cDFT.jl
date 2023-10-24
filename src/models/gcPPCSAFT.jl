using Clapeyron: gcPPCSAFTModel

function F_res(model::gcPPCSAFTModel,ρ,T,z)
    return F_res(model.ppcmodel,ρ,T,z)
end

function δFδρ_res(model::gcPPCSAFTModel,ρ,T,z)
    return δFδρ_res(model.ppcmodel,ρ,T,z)
end

function length_scale(model::gcPPCSAFTModel)
    return length_scale(model.ppcmodel)
end