struct SteeleParam <: EoSParam
    sigma::SingleParam{Float64}
    epsilon::SingleParam{Float64}
    delta::SingleParam{Float64}
    rho::SingleParam{Float64}
    width::SingleParam{Float64}
end

abstract type SteeleModel <: ExternalFieldModel end

struct Steele <: SteeleModel
    surface::Array{String,1}
    params::SteeleParam
    references::Array{String,1}
end

export Steele

function Steele(surface::Array{String,1}, width::Float64=0.0; userlocations = String[], verbose = false)
    params = getparams(surface, ["$DB_PATH/Steele/"]; userlocations = userlocations, verbose = verbose)
    references = ["10.1016/0039-6028(73)90264-1"]
    ϵs = params["epsilon"]
    σs = params["sigma"]
    σs.values .*= 1e-10

    Δ = params["delta"]
    Δ.values .*= 1e-10
    ρ = params["rho"]
    ρ.values .*= 1e30
    width_param = SingleParam("width", surface, [width])

    packagedparams = SteeleParam(σs,ϵs,Δ,ρ,width_param)
    return Steele(surface, packagedparams, references)
end


function evaluate_external_field!(structure::DFTStructure,external_field::SteeleModel,model::SAFTModel,ρ,δfδρ_res,z,Vext=nothing)

    if Vext == nothing
        Vext = zeros(eltype(ρ), size(ρ))
        nd = dimension(structure)
        (_,temperature) = structure.conditions
        ϵs = external_field.params.epsilon.values
        σs = external_field.params.sigma.values
        Δ  = external_field.params.delta.values
        ρ  = external_field.params.rho.values
        H  = external_field.params.width.values

        ϵi = diagvalues(model.params.epsilon.values)
        σi = diagvalues(model.params.sigma.values)
        
        ngrid = structure.ngrid
        nbeads = length(ϵi)
        nsurf = length(ϵs)
        for s in 1:nsurf
            ϵsi = sqrt.(ϵs[s].*ϵi)
            σsi = (σs[s].+σi)/2
            for i in 1:nbeads
                selectdim(Vext,nd+1,i) .+= @. 2π*ρ[s]*ϵsi[i]*Δ[s]*σsi[i]^2*(2/5*(σsi[i]/z)^10-(σsi[i]/z)^4-σsi[i]^4/(3*Δ[s]*(z+0.61*Δ[s])^3))
                if H!=0
                    selectdim(Vext,nd+1,i) .+= @. 2π*ρ[s]*ϵsi[i]*Δ[s]*σsi[i]^2*(2/5*(σsi[i]/(H-z))^10-(σsi[i]/(H-z))^4-σsi[i]^4/(3*Δ[s]*((H-z)+0.61*Δ[s])^3))
                end
            end
        end
        return Vext .*= 1/temperature
    else
        δfδρ_res .+= Vext
    end
end

function evaluate_external_field!(structure::DFTStructure,external_field::SteeleModel,model::ElectrolyteModel,ρ,δfδρ_res,z,Vext=nothing)
    evaluate_external_field!(structure,external_field,model.neutralmodel,ρ,δfδρ_res,z, Vext)
end

"""
    evaluate_external_field!(structure::Union{DFTStructureSphr,DFTStructureCyl}, external_field::SteeleModel, model::SAFTModel, ρ, δfδρ_res, r, Vext=nothing)

Spherical/cylindrical counterpart of the planar Steele wall above, reusing the same
single-wall LJ-9-3 formula. Since the QDHT-based radial grid always spans from ~0 to
`bounds[2]` (see `Uniform1DSphr`/`Uniform1DCyl`), the wall is placed via `bounds`
rather than by domain truncation:
- `bounds[1] > 0`: fluid *outside* a solid sphere/cylinder of radius `bounds[1]` — wall
  distance is `r - bounds[1]`.
- `bounds[1] ≈ 0`: fluid *inside* a pore of radius `bounds[2]` — wall distance is
  `bounds[2] - r`.

The wall-distance is clamped at `0.5*minimum(σi)` (matching the margin convention
already used by the planar `ExternalField1DCart`/`TwoPhase1DCart` structures, whose
`bounds` are similarly kept `0.5L` clear of the wall) — both to avoid the LJ-9-3
formula's literal divergence at contact, and, importantly, so the wall potential
(and hence `exp(-Vext/10)`) stays finite rather than underflowing to an exact `0.0`
density: an exact-zero density then causes `log`/association terms elsewhere in the
free energy functional to produce `NaN`, which spreads to the *entire* profile after
one convolution pass. The clamp is applied uniformly for `r` on both sides of the wall
(rather than a separate hard-exclusion branch with an arbitrary large constant), so the
potential stays smooth and bounded throughout, including deep inside the excluded
region.
"""
function evaluate_external_field!(structure::Union{DFTStructureSphr,DFTStructureCyl},external_field::SteeleModel,model::SAFTModel,ρ,δfδρ_res,r,Vext=nothing)
    if Vext === nothing
        Vext = zeros(eltype(ρ), size(ρ))
        nd = dimension(structure)
        (_,temperature) = structure.conditions
        ϵs = external_field.params.epsilon.values
        σs = external_field.params.sigma.values
        Δ  = external_field.params.delta.values
        ρwall = external_field.params.rho.values

        ϵi = diagvalues(model.params.epsilon.values)
        σi = diagvalues(model.params.sigma.values)

        lb, ub = bounds(structure, 1)
        zwall_raw = lb > 0 ? (r .- lb) : (ub .- r)
        zwall     = max.(zwall_raw, 0.5*minimum(σi))

        nbeads = length(ϵi)
        nsurf = length(ϵs)
        for s in 1:nsurf
            ϵsi = sqrt.(ϵs[s].*ϵi)
            σsi = (σs[s].+σi)/2
            for i in 1:nbeads
                selectdim(Vext,nd+1,i) .+= @. 2π*ρwall[s]*ϵsi[i]*Δ[s]*σsi[i]^2*(2/5*(σsi[i]/zwall)^10-(σsi[i]/zwall)^4-σsi[i]^4/(3*Δ[s]*(zwall+0.61*Δ[s])^3))
            end
        end
        Vext .*= 1/temperature

        return Vext
    else
        δfδρ_res .+= Vext
    end
end

function evaluate_external_field!(structure::Union{DFTStructureSphr,DFTStructureCyl},external_field::SteeleModel,model::ElectrolyteModel,ρ,δfδρ_res,r,Vext=nothing)
    evaluate_external_field!(structure,external_field,model.neutralmodel,ρ,δfδρ_res,r, Vext)
end