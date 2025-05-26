struct ConstGradientParam <: EoSParam
    kappa::PairParam{Float64}
end

abstract type ConstGradientModel <: GradientModel end

struct ConstGradient <: ConstGradientModel
    species::Array{String,1}
    params::ConstGradientParam
    references::Array{String,1}
end

export ConstGradient

function ConstGradient(components::Array{String,1}; 
                            userlocations = String[],
                            verbose = false)
    params = getparams(components, ["$DB_PATH/gradients/"]; userlocations = userlocations, verbose = verbose)
    references = String[]
    κ = params["kappa"]
    β = get(params,"beta",nothing)
    κij = epsilon_LorentzBerthelot(κ,β)
    packagedparams = ConstGradientParam(κij)
    return ConstGradient(components, packagedparams, references)
end


function kappa(gradient::ConstGradientModel, T, ρ̄)
    return gradient.params.kappa.values
end