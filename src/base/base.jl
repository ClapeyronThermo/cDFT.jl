abstract type DFTProfile end
abstract type DFTField end
abstract type DFTSpecies end
abstract type DFTPropagator end

include("devices.jl")
include("structure.jl")

"""
    DFTSystem(model::EoSModel, species::DFTSpecies, structure::DFTStructure, profiles::Vector{DFTProfile}, fields::Vector{DFTField}, options::DFTOptions)

Generic struct which includes all the information needed to perform DFT / SCFT calculations:
- `model`: A model object that should be obtained from Clapeyron.jl, and contains all information regarding species parameters.
- `species`: A `DFTSpecies` object which is model-dependent. Typically contains the number of beads in each species and the bead sizes.
- `structure`: A `DFTStructure` object which provides information regarding the geometry (including bounds) and conditions (n, p, T) of the DFT calculations.
- `profiles`: A vector of `DFTProfile`s for each species / bead in the system. By default, these will be cubic splines.
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
struct DFTSystem
    model::EoSModel
    species::Vector{DFTSpecies}
    structure::DFTStructure
    profiles::Vector{DFTProfile}
    fields::Vector{DFTField}
    propagator::DFTPropagator
    options::DFTOptions
end

function DFTSystem(model::EoSModel,structure::DFTStructure,profile::Vector{DFTProfile},fields::Vector{DFTField},options::DFTOptions)
    return DFTSystem(model,species,structure,profile,fields,options)
end

function DFTSystem(model::EoSModel, structure::DFTStructure, options::DFTOptions = DFTOptions())
    species = get_species(model, structure)
    fields = get_fields(model)
    propagator = get_propagator(model)
    profiles = initialize_profiles(model,structure, species)
    return DFTSystem(model, species, structure, profiles, fields, propagator, options)
end

export DFTSystem

include("show.jl")