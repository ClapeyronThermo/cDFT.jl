struct ConstGradientParam <: EoSParam
    kappa::PairParam{Float64}
end

abstract type ConstGradientModel <: GradientModel end

"""
    ConstGradient <: GradientModel

`GradientModel` for use with `DGTSystem`, in which the influence parameter `kappa` of the square-gradient term is a constant (density- and temperature-independent) pair matrix, obtained via combining rule from single-species values in the parameter database.
"""
struct ConstGradient <: ConstGradientModel
    species::Array{String,1}
    params::ConstGradientParam
    references::Array{String,1}
end

export ConstGradient

"""
    ConstGradient(components::Array{String,1}; userlocations = String[], verbose = false)

Construct a `ConstGradient` gradient model for the given `components`, looking up single-species influence parameters from the gradient parameter database (or `userlocations` if provided) and combining them into a pair matrix via the Lorentz-Berthelot-style combining rule.
"""
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