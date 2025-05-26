abstract type DFTProfile end
abstract type DFTField end
abstract type DFTSpecies end
abstract type DFTPropagator end
abstract type ExternalFieldModel end
abstract type GradientModel end

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
struct DFTSystem{M<:EoSModel,S<:DFTSpecies,T<:DFTStructure,F,P<:DFTPropagator,O<:DFTOptions,C}
    model::M
    species::S
    structure::T
    fields::F
    propagator::P
    options::O
    chunksize::Val{C}
end

function DFTSystem(model::EoSModel, structure::DFTStructure, options::DFTOptions = DFTOptions())
    species = get_species(model, structure)
    fields = get_fields(model, species, structure)
    typed_fields = tuple(fields...)
    propagator = get_propagator(model, species, structure)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, typed_fields, propagator, options,chunksize)
end

"""
    DGTSystem(model::EoSModel, species::DFTSpecies, structure::DFTStructure, fields::Vector{DFTField}, options::DFTOptions)

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
struct DGTSystem{M<:EoSModel,S<:DFTSpecies,T<:DFTStructure,F,G<:GradientModel,O<:DFTOptions,C}
    model::M
    gradient::G
    species::S
    structure::T
    fields::F
    options::O
    chunksize::Val{C}
end

function DGTSystem(model::EoSModel, gradient::GradientModel, structure::DFTStructure, options::DFTOptions = DFTOptions())
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    ω = structure_ω(structure)

    nc = length(model)
    sizes = length_scales(model)
    nbeads = ones(Int64,nc)
    
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T

    species = DGTSpecies(nbeads,sizes,ρbulk,μres)
    fields = [SWeightedDensity(:ρ,zeros(nc),ω,ngrid),
              VWeightedDensity(:∇ρ,zeros(nc),ω,ngrid)]
    typed_fields = tuple(fields...)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DGTSystem(model, gradient, species, structure, typed_fields, options, chunksize)
end

const AbstractcDFTSystem = AbstractcDFTSystem

dimension(::Type{Union{DFTSystem{<:Any,<:Any,T},DGTSystem{<:Any,<:Any,T}}}) where T = dimension(T) 
dimension(x::AbstractcDFTSystem) = dimension(x.structure)

length_fields(system::AbstractcDFTSystem) = length_fields(system.chunksize)
length_fields(::ForwardDiff.Chunk{N}) where N = N
length_fields(::Val{N}) where N = N

function compute_field_len(fields,nd)
    field_len = 0
    for field in fields
        if field isa ScalarField
            field_len += 1
        elseif field isa VectorField
            field_len += nd
        end
    end
    return field_len
end

ForwardDiff.Chunk(system::T) where T <: AbstractcDFTSystem = FDChunk(system.chunksize)
FDChunk(system::T) where T <: AbstractcDFTSystem = FDChunk(system.chunksize)
FDChunk(::Val{N}) where N = ForwardDiff.Chunk{N}(N)

export DFTSystem, DGTSystem

include("show.jl")