abstract type DFTProfile end
abstract type DFTField end
abstract type DFTSpecies end
abstract type DFTPropagator end
abstract type ExternalFieldModel end
abstract type GradientModel end

"""
    AbstractcDFTSystem

Supertype for every system cDFT.jl can converge via `converge!`: `DFTSystem`,
`DGTSystem`, `ElectrolyteDFTSystem` (all defined below), and `SCFTSystem`
(`src/models/SCFT/scft.jl`, loaded much later via `models/models.jl` — a real abstract
type is required here, rather than a `Union`, since a `Union` literal can't forward-
reference a type that doesn't exist yet at this point in the load order).
"""
abstract type AbstractcDFTSystem end

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
struct DFTSystem{M<:EoSModel,S<:DFTSpecies,T<:DFTStructure,F,EF,P<:DFTPropagator,O<:DFTOptions,C} <: AbstractcDFTSystem
    model::M
    species::S
    structure::T
    fields::F
    external_field::EF
    propagator::P
    options::O
    chunksize::Val{C}
end

function build_DFT_system(orig_model,structure,external_field,options,mol_structure)
    model = expand_model(orig_model, mol_structure) #if the model has no groups, it does nothing
    species = get_species(model, structure)
    FP = fptype(options)
    device = options.device
    fields = get_fields(model, species, structure, device, FP)
    propagator = get_propagator(model, species, structure, device, FP)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    init_external_field = external_field isa ExternalFieldModel ? [external_field] : external_field
    return DFTSystem(model, species, structure, fields, init_external_field, propagator, options, chunksize)
end


function DFTSystem(model::EoSModel, structure::DFTStructure; mol_structure = nothing)
    return build_DFT_system(model,structure,nothing,DFTOptions(),mol_structure)
end

function DFTSystem(model::EoSModel, structure::DFTStructure, options::DFTOptions; mol_structure = nothing)
    return build_DFT_system(model,structure,nothing,options,mol_structure)
end

function DFTSystem(model::EoSModel, structure::DFTStructure, external_field::Union{ExternalFieldModel,Vector{<:ExternalFieldModel},Nothing}; mol_structure = nothing)
    build_DFT_system(model,structure,external_field,DFTOptions(),mol_structure)
end

function DFTSystem(model::EoSModel, structure::DFTStructure, external_field::Union{ExternalFieldModel,Vector{<:ExternalFieldModel},Nothing},options::DFTOptions; mol_structure = nothing)
    build_DFT_system(model,structure,external_field,options,mol_structure)
end

get_fields(model, species, structure, device) = get_fields(model, species, structure, device, Float64)

"""
    DGTSystem(model::EoSModel, gradient::GradientModel, structure::DFTStructure, external_field, options::DFTOptions)

Generic struct which includes all the information needed to perform Density Gradient Theory (DGT/SCFT) calculations, i.e. a square-gradient functional built on top of a bulk `model` rather than a full weighted-density DFT functional:
- `model`: A model object that should be obtained from Clapeyron.jl, and contains all information regarding species parameters.
- `gradient`: A `GradientModel` (e.g. `ConstGradient`) which supplies the influence parameter(s) `κ` for the square-gradient term.
- `species`: A `DGTSpecies` object containing the bulk densities, chemical potentials and length scales for each species.
- `structure`: A `DFTStructure` object which provides information regarding the geometry (including bounds) and conditions (n, p, T) of the DFT calculations.
- `fields`: A vector of `DFTField`s for each field used in the calculation (density and its gradient).
- `options`: A `DFTOptions` object which contains information regarding the convergence settings and the devices used as part of the calculation.
Example usage:
```julia
julia> model = PCSAFT(["water"])

julia> gradient = ConstGradient(["water"])

julia> L = length_scale(model)

julia> structure = Uniform1DCart((1e5, 298.15, [1.]), [0, 20L], 201)

julia> system = DGTSystem(model, gradient, structure)
```
"""
struct DGTSystem{M<:EoSModel,S<:DFTSpecies,T<:DFTStructure,F,EF,G<:GradientModel,O<:DFTOptions,C} <: AbstractcDFTSystem
    model::M
    gradient::G
    species::S
    structure::T
    fields::F
    external_field::EF
    options::O
    chunksize::Val{C}
end

function DGTSystem(model::EoSModel, gradient::GradientModel, structure::DFTStructure, external_field::ExternalFieldModel, options::DFTOptions = DFTOptions())
    backend = options.device
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    ω = structure_ω(structure, backend, fptype(options))

    nc = length(model)
    sizes = length_scales(model)
    nbeads = ones(Int64,nc)
    
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature

    species = DGTSpecies(nbeads,sizes,ρbulk,μres)
    # :ρ/:∇ρ kernels don't depend on `width` at all (Ω=1 and Ω=i*2π*ω respectively — no
    # smoothing/weighting shape to rescale, unlike the SAFT-family's convolution kernels).
    # `model` is threaded through so the constructor can compute `L=length_scale(model)`
    # internally for `density_scale`/`NA` (`:∇ρ` deliberately keeps `ω` raw — see
    # VWeightedDensity's docstring). See dgt.jl's `f_res` for how `preallocate_params`
    # compensates (κ/L^3, V=N_A*L^3 into a_res) to keep the whole f_res output uniformly
    # L^3-inflated, matching `_energy_scale(::DGTSystem)`.
    fields = [SWeightedDensity(:ρ,zeros(nc),ω,ngrid,backend,model),
              VWeightedDensity(:∇ρ,zeros(nc),ω,ngrid,backend,model)]
    typed_fields = tuple(fields...)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DGTSystem(model, gradient, species, structure, typed_fields, [external_field], options, chunksize)
end

function DGTSystem(model::EoSModel, gradient::GradientModel, structure::DFTStructure, external_field::Vector{ExternalFieldModel}, options::DFTOptions = DFTOptions())
    backend = options.device
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    ω = structure_ω(structure, backend, fptype(options))

    nc = length(model)
    sizes = length_scales(model)
    nbeads = ones(Int64,nc)

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature

    species = DGTSpecies(nbeads,sizes,ρbulk,μres)
    # :ρ/:∇ρ kernels don't depend on `width` at all (Ω=1 and Ω=i*2π*ω respectively — no
    # smoothing/weighting shape to rescale, unlike the SAFT-family's convolution kernels).
    # `model` is threaded through so the constructor can compute `L=length_scale(model)`
    # internally for `density_scale`/`NA` (`:∇ρ` deliberately keeps `ω` raw — see
    # VWeightedDensity's docstring). See dgt.jl's `f_res` for how `preallocate_params`
    # compensates (κ/L^3, V=N_A*L^3 into a_res) to keep the whole f_res output uniformly
    # L^3-inflated, matching `_energy_scale(::DGTSystem)`.
    fields = [SWeightedDensity(:ρ,zeros(nc),ω,ngrid,backend,model),
              VWeightedDensity(:∇ρ,zeros(nc),ω,ngrid,backend,model)]
    typed_fields = tuple(fields...)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DGTSystem(model, gradient, species, structure, typed_fields, external_field, options,chunksize)
end

function DGTSystem(model::EoSModel, gradient::GradientModel, structure::DFTStructure, options::DFTOptions = DFTOptions())
    backend = options.device
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    ω = structure_ω(structure, backend, fptype(options))

    nc = length(model)
    sizes = length_scales(model)
    nbeads = ones(Int64,nc)

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature

    species = DGTSpecies(nbeads,sizes,ρbulk,μres)
    # :ρ/:∇ρ kernels don't depend on `width` at all (Ω=1 and Ω=i*2π*ω respectively — no
    # smoothing/weighting shape to rescale, unlike the SAFT-family's convolution kernels).
    # `model` is threaded through so the constructor can compute `L=length_scale(model)`
    # internally for `density_scale`/`NA` (`:∇ρ` deliberately keeps `ω` raw — see
    # VWeightedDensity's docstring). See dgt.jl's `f_res` for how `preallocate_params`
    # compensates (κ/L^3, V=N_A*L^3 into a_res) to keep the whole f_res output uniformly
    # L^3-inflated, matching `_energy_scale(::DGTSystem)`.
    fields = [SWeightedDensity(:ρ,zeros(nc),ω,ngrid,backend,model),
              VWeightedDensity(:∇ρ,zeros(nc),ω,ngrid,backend,model)]
    typed_fields = tuple(fields...)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DGTSystem(model, gradient, species, structure, typed_fields, nothing, options,chunksize)
end

"""
    ElectrolyteDFTSystem(model::ElectrolyteModel, structure::DFTStructure, external_field, options::DFTOptions)

Generic struct which includes all the information needed to perform DFT / SCFT calculations for electrolyte systems:
- `model`: An `ElectrolyteModel` obtained from Clapeyron.jl, combining a neutral bulk model (`model.neutralmodel`) with an ion model (`model.ionmodel`) and the species charges.
- `species`: A `DFTSpecies` object for the neutral species, obtained from `model.neutralmodel`.
- `ion_species`: A `DFTSpecies` object for the ionic species, obtained from `model.ionmodel`.
- `structure`: A `DFTStructure` object which provides information regarding the geometry (including bounds) and conditions (n, p, T) of the DFT calculations.
- `fields`: A vector of `DFTField`s for the neutral and ionic species combined.
- `external_field`: In addition to any user-supplied external field(s), an `ElectrostaticPotential` field is always appended automatically to account for the mean-field electrostatic interactions between ions.
- `propagator`: The `DFTPropagator` used for the neutral model (ions use an `IdealPropagator`).
- `options`: A `DFTOptions` object which contains information regarding the convergence settings and the devices used as part of the DFT calculation.
Example usage:
```julia
julia> model = SPCSAFT(["water"]) + DH(["water"],["Na","Cl"])

julia> L = length_scale(model)

julia> structure = Uniform1DCart((1e5, 298.15, [1.]), [0, 20L], 201)

julia> system = ElectrolyteDFTSystem(model, structure)
```
"""
struct ElectrolyteDFTSystem{M<:ElectrolyteModel,S<:DFTSpecies,iS<:DFTSpecies,T<:DFTStructure,F,EF,P<:DFTPropagator,O<:DFTOptions,C} <: AbstractcDFTSystem
    model::M
    species::S
    ion_species::iS
    structure::T
    fields::F
    external_field::EF
    propagator::P
    options::O
    chunksize::Val{C}
end

dimension(::Type{Union{DFTSystem{<:Any,<:Any,T},DGTSystem{<:Any,<:Any,T}}}) where T = dimension(T)
dimension(x::AbstractcDFTSystem) = dimension(x.structure)

length_fields(system::AbstractcDFTSystem) = length_fields(system.chunksize)
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

export DFTSystem, DGTSystem, ElectrolyteDFTSystem

include("show.jl")