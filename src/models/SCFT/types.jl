"""
    SCFTInteraction

Abstract type for SCFT interaction models. Subtypes define how fields are
computed from densities.
"""
abstract type SCFTInteraction end

"""
    FloryHuggins(chi, rho0, kappa)

Local Flory-Huggins interaction model with Helfand compressibility.

# Fields
- `chi::Matrix{Float64}`: χ_{αβ} interaction parameters (symmetric, zero diagonal).
- `rho0::Float64`: Total reference density ρ₀.
- `kappa::Float64`: Helfand compressibility parameter ζ.
"""
struct FloryHuggins <: SCFTInteraction
    chi::Matrix{Float64}
    rho0::Float64
    kappa::Float64
    function FloryHuggins(chi::Matrix{Float64}, rho0::Float64, kappa::Float64)
        @assert size(chi, 1) == size(chi, 2) "chi must be square"
        @assert issymmetric(chi) "chi must be symmetric"
        @assert all(diag(chi) .== 0) "chi diagonal must be zero"
        new(chi, rho0, kappa)
    end
end

"""
    SCFTChain(; N, b, segment_species, ensemble, n_chains=0.0, bulk_density=0.0)

Definition of a polymer chain type for SCFT.

# Fields
- `N::Int`: Number of segments.
- `b::Float64`: Statistical segment length.
- `segment_species::Vector{Int}`: Segment index → species index mapping.
- `ensemble::Symbol`: `:canonical` or `:grand_canonical`.
- `n_chains::Float64`: Number of chains (canonical ensemble).
- `bulk_density::Float64`: Bulk segment density φ⁰ (grand canonical ensemble).
"""
struct SCFTChain
    N::Int
    b::Float64
    segment_species::Vector{Int}
    ensemble::Symbol
    n_chains::Float64
    bulk_density::Float64
    function SCFTChain(; N::Int, b::Float64, segment_species::Vector{Int},
                        ensemble::Symbol, n_chains::Float64=0.0, bulk_density::Float64=0.0)
        @assert length(segment_species) == N "segment_species length must equal N"
        @assert ensemble in (:canonical, :grand_canonical) "ensemble must be :canonical or :grand_canonical"
        if ensemble == :canonical
            @assert n_chains > 0 "n_chains must be positive for canonical ensemble"
        else
            @assert bulk_density > 0 "bulk_density must be positive for grand canonical ensemble"
        end
        new(N, b, segment_species, ensemble, n_chains, bulk_density)
    end
end

"""
    SCFTSolvent(; species_index, ensemble, n_molecules=0.0, bulk_density=0.0)

Monomeric solvent species for SCFT.

# Fields
- `species_index::Int`: Index of the solvent species.
- `ensemble::Symbol`: `:canonical` or `:grand_canonical`.
- `n_molecules::Float64`: Number of molecules (canonical).
- `bulk_density::Float64`: Bulk density ρ_S^b (grand canonical).
"""
struct SCFTSolvent
    species_index::Int
    ensemble::Symbol
    n_molecules::Float64
    bulk_density::Float64
    function SCFTSolvent(; species_index::Int, ensemble::Symbol,
                          n_molecules::Float64=0.0, bulk_density::Float64=0.0)
        @assert ensemble in (:canonical, :grand_canonical) "ensemble must be :canonical or :grand_canonical"
        if ensemble == :canonical
            @assert n_molecules > 0 "n_molecules must be positive for canonical ensemble"
        else
            @assert bulk_density > 0 "bulk_density must be positive for grand canonical ensemble"
        end
        new(species_index, ensemble, n_molecules, bulk_density)
    end
end

"""
    SCFTSystem(; interaction, chains, solvents, nspecies, species_names, bounds, ngrid, options)

Main SCFT system struct containing all information for SCFT calculations.

# Fields
- `interaction::SCFTInteraction`: Interaction model (e.g., FloryHuggins).
- `structure::DFTStructure`: Spatial discretization.
- `propagator::DFTPropagator`: Chain propagator (DiscreteGaussianChainPropagator).
- `chains::Vector{SCFTChain}`: Chain type definitions.
- `solvents::Vector{SCFTSolvent}`: Solvent definitions.
- `nspecies::Int`: Total number of species types.
- `species_names::Vector{Symbol}`: Names for each species.
- `options::DFTOptions`: Device and solver options.
"""
struct SCFTSystem{I<:SCFTInteraction, T<:DFTStructure, P<:DFTPropagator, O<:DFTOptions}
    interaction::I
    structure::T
    propagator::P
    chains::Vector{SCFTChain}
    solvents::Vector{SCFTSolvent}
    nspecies::Int
    species_names::Vector{Symbol}
    options::O
end

dimension(x::SCFTSystem) = dimension(x.structure)

"""
    SCFTSystem(; interaction, chains, solvents=SCFTSolvent[], nspecies, species_names,
                 bounds, ngrid, options=DFTOptions())

Construct an SCFTSystem. Automatically builds the spatial structure and
chain propagator from the provided parameters.

# Arguments
- `interaction`: An `SCFTInteraction` (e.g., `FloryHuggins`).
- `chains`: Vector of `SCFTChain` definitions.
- `solvents`: Vector of `SCFTSolvent` definitions (default: empty).
- `nspecies`: Total number of species types.
- `species_names`: Symbol names for each species.
- `bounds`: Domain bounds — `Vector{Float64}` for 1D, `Matrix{Float64}` for 2D/3D.
- `ngrid`: Grid points — `Int` for 1D (or uniform 3D), `Tuple` for anisotropic.
- `options`: `DFTOptions` for device/solver settings.
"""
function SCFTSystem(;
    interaction::SCFTInteraction,
    chains::Vector{SCFTChain},
    solvents::Vector{SCFTSolvent} = SCFTSolvent[],
    nspecies::Int,
    species_names::Vector{Symbol},
    bounds,
    ngrid,
    options::DFTOptions = DFTOptions()
)
    # Build b_species: each species gets its b from the chains that use it
    b_species = zeros(nspecies)
    b_set = falses(nspecies)
    for chain in chains
        for (s, sp) in enumerate(chain.segment_species)
            if !b_set[sp]
                b_species[sp] = chain.b
                b_set[sp] = true
            end
        end
    end

    # Dummy bulk densities for structure construction
    dummy_ρbulk = ones(nspecies)

    # Build structure based on dimensionality of bounds
    structure = if bounds isa Vector{Float64} || bounds isa Vector
        Uniform1DCart((0.0, 0.0), dummy_ρbulk, Float64.(bounds), ngrid isa Int ? ngrid : ngrid[1])
    elseif bounds isa Matrix{Float64} || bounds isa Matrix
        nd = size(bounds, 1)
        if nd == 2
            ng = ngrid isa Int ? (ngrid, ngrid) : Tuple(ngrid[1:2])
            Uniform2DCart((0.0, 0.0), dummy_ρbulk, Float64.(bounds), ng)
        elseif nd == 3
            Uniform3DCart((0.0, 0.0), dummy_ρbulk, Float64.(bounds), ngrid isa Int ? ngrid : ngrid[1])
        else
            error("Unsupported number of dimensions: $nd (must be 1, 2, or 3)")
        end
    else
        error("Unsupported bounds type: $(typeof(bounds))")
    end

    # Build propagator
    N_vec = [c.N for c in chains]
    seg_spec_vec = [c.segment_species for c in chains]
    propagator = DiscreteGaussianChainPropagator(b_species, N_vec, seg_spec_vec, structure, options.device)

    return SCFTSystem(interaction, structure, propagator, chains, solvents, nspecies, species_names, options)
end
