abstract type DFTProfile end
abstract type DFTField end
abstract type DFTSpecies end
abstract type DFTPropagator end
abstract type ExternalFieldModel end

const DB_PATH = normpath(Base.pkgdir(cDFT),"database")

include("devices.jl")
include("structure.jl")

"""
    DFTSystem(model::EoSModel, species::DFTSpecies, structure::DFTStructure, fields::Vector{DFTField}, options::DFTOptions)

Generic struct which includes all the information needed to perform DFT / SCFT calculations:
- `model`: A model object that should be obtained from Clapeyron.jl, and contains all information regarding species parameters.
- `species`: A `DFTSpecies` object which is model-dependent. Typically contains the number of beads in each species and the bead sizes.
- `structure`: A `DFTStructure` object which provides information regarding the geometry (including bounds) and conditions (n, p, T) of the DFT calculations.
- `fields`: A vector of `DFTField`s for each field used in the DFT-calculation. This is typically model-dependent. 
- `options`: A `DFTOptions` object which contains information regarding the convergence settings and the devices used as part of the DFT calculation.
Example usage:
```julia
julia> model = PCSAFT(["water"])

julia> L = length_scale(model)

julia> structure = Uniform1DCart((1e5, 298.15, [1.]), [0, 20L], 201)

julia> system = DFTSystem(model, structure)
DFTSystem
  model: PCSAFT{BasicIdeal, Float64}
         with 1 component: "water"
  structure: Uniform1DCart
  device: CPU
```
"""
struct DFTSystem{M<:EoSModel,S<:DFTSpecies,T<:DFTStructure,F<:DFTField,P<:DFTPropagator,O<:DFTOptions}
    model::M
    species::S
    structure::T
    fields::Vector{F}
    propagator::P
    options::O
end

dimension(::Type{DFTSystem{<:Any,<:Any,T}}) where T = dimension(T) 

function DFTSystem(model::EoSModel,structure::DFTStructure,profile::Vector{DFTProfile},fields::Vector{DFTField},options::DFTOptions)
    return DFTSystem(model,species,structure,profile,fields,options)
end

function DFTSystem(model::EoSModel, structure::DFTStructure, options::DFTOptions = DFTOptions())
    species = get_species(model, structure)
    fields = get_fields(model, species, structure)
    propagator = get_propagator(model, species, structure)
    return DFTSystem(model, species, structure, fields, propagator, options)
end

# length_fields(system::DFTSystem) = length_fields(system.model)

function length_fields(system::DFTSystem)
    fields = system.fields
    nd = dimension(system)
    nfscalar = sum(typeof.(fields) .<: ScalarField)
    nfvector = sum(typeof.(fields) .<: VectorField)
    return nfscalar + nfvector*nd
end

dimension(x::DFTSystem) = dimension(x.structure)

export DFTSystem

include("show.jl")