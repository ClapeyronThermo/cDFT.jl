struct SteeleParam <: EoSParam
    sigma::SingleParam{Float64}
    epsilon::SingleParam{Float64}
    delta::SingleParam{Float64}
    rho::SingleParam{Float64}
end

abstract type SteeleModel <: ExternalFieldModel end

struct Steele <: SteeleModel
    surface::Array{String,1}
    params::SteeleParam
    references::Array{String,1}
end

export Steele

function Steele(surface::Array{String,1}; 
                            userlocations = String[],
                            verbose = false)
    params = getparams(surface, ["$DB_PATH/Steele/"]; userlocations = userlocations, verbose = verbose)
    references = ["10.1016/0039-6028(73)90264-1"]
    ϵs = params["epsilon"]
    σs = params["sigma"]
    σs.values .*= 1e-10

    Δ = params["delta"]
    Δ.values .*= 1e-10
    ρ = params["rho"]
    ρ.values .*= 1e30

    packagedparams = SteeleParam(σs,ϵs,Δ,ρ)
    return Steele(surface, packagedparams, references)
end


function evaluate_external_field(structure::DFTStructure,external_field::SteeleModel,model::SAFTModel,z)
    nd = dimension(structure)
    H = structure.width
    (_,T) = structure.conditions
    ϵs = external_field.params.epsilon.values
    σs = external_field.params.sigma.values
    Δ  = external_field.params.delta.values
    ρ  = external_field.params.rho.values

    ϵi = diagvalues(model.params.epsilon.values)
    σi = diagvalues(model.params.sigma.values)
    
    ngrid = structure.ngrid
    nbeads = length(ϵi)
    nsurf = length(ϵs)
    external_field_values = zeros(ngrid...,nbeads)
    for s in 1:nsurf
        ϵsi = sqrt.(ϵs[s].*ϵi)
        σsi = (σs[s].+σi)/2
        for i in 1:nbeads
            selectdim(external_field_values,nd+1,i) .+= @. 2π*ρ[s]*ϵsi[i]*Δ[s]*σsi[i]^2*(2/5*(σsi[i]/z)^10-(σsi[i]/z)^4-σsi[i]^4/(3*Δ[s]*(z+0.61*Δ[s])^3))
            if H!=0
                selectdim(external_field_values,nd+1,i) .+= @. 2π*ρ[s]*ϵsi[i]*Δ[s]*σsi[i]^2*(2/5*(σsi[i]/(H-z))^10-(σsi[i]/(H-z))^4-σsi[i]^4/(3*Δ[s]*((H-z)+0.61*Δ[s])^3))
            end
        end
    end
    external_field_values = external_field_values./T 
    external_field_values[external_field_values .> 100] .= 100
    return external_field_values
end

function evaluate_external_field(structure::DFTStructure,external_field::SteeleModel,model::SAFTModel,ρ::Array{Float64},z)
    return evaluate_external_field(structure,external_field,model,z)
end