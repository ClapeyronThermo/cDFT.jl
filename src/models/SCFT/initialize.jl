"""
    initialize_profiles(system::SCFTSystem; mode=:uniform, perturbation=0.01)

Initialize density profiles for an SCFT system.

# Arguments
- `mode::Symbol`: `:uniform` for flat bulk profiles, `:perturbed` for bulk + random noise.
- `perturbation::Float64`: Amplitude of random perturbation (relative to bulk density).

# Returns
Density array of size `(ngrid..., nspecies)`.
"""
function initialize_profiles(system::SCFTSystem; mode::Symbol=:uniform, perturbation::Float64=0.01)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    nspecies = system.nspecies
    device = system.options.device

    FT = float_type(system.options)

    # Compute bulk densities per species (always Float64 internally, then convert)
    bulk = FT.(compute_bulk_densities(system))

    ρ = allocate(device, FT, ngrid..., nspecies)

    for α in 1:nspecies
        selectdim(ρ, nd+1, α) .= bulk[α]
    end

    if mode == :perturbed
        for α in 1:nspecies
            noise_cpu = FT(perturbation) * bulk[α] .* (FT(2) .* rand(FT, ngrid...) .- FT(1))
            selectdim(ρ, nd+1, α) .+= Adapt.adapt(device, noise_cpu)
            # Ensure non-negative
            clamp!(selectdim(ρ, nd+1, α), FT(1e-10), FT(Inf))
        end
        # Normalize to maintain incompressibility: sum_α ρ_α = rho0 at each point.
        # Without this, independent noise on each species causes total density to deviate
        # from rho0, producing huge compressibility fields (O(kappa × noise)) that make
        # the Picard warmup diverge.
        rho0 = FT(system.interaction.rho0)
        ρ_total = sum(ρ, dims=nd+1)  # shape: (ngrid..., 1)
        ρ .*= rho0 ./ ρ_total
    end

    return ρ
end

"""
    compute_bulk_densities(system::SCFTSystem; V_eff=nothing)

Compute the bulk density of each species from the chain and solvent definitions.

# Keyword Arguments
- `V_eff`: Effective domain volume to use for canonical-ensemble prefactors
  (`n_chains / V_eff`). When `nothing`, falls back to `effective_volume(system, dz)`
  (CPU Simpson rule). Pass the same `V_eff` used in the SCFT iteration loop to
  ensure the bulk densities are consistent with the chosen quadrature rule.

Returns a vector of length `nspecies` with element type `float_type(system.options)`.
"""
function compute_bulk_densities(system::SCFTSystem; V_eff=nothing)
    nspecies = system.nspecies
    FT = float_type(system.options)
    bulk = zeros(FT, nspecies)
    dz = structure_dz(system.structure)
    V_eff = V_eff !== nothing ? FT(V_eff) : FT(effective_volume(system, dz))

    for chain in system.chains
        seg_spec = chain.segment_species
        N = chain.N

        if chain.ensemble == :canonical
            chain_density = FT(chain.n_chains) / V_eff
            for s in 1:N
                α = seg_spec[s]
                bulk[α] += chain_density
            end
        else
            # Grand canonical: bulk_density is the segment density φ⁰
            # Each segment contributes φ⁰/N to its species
            chain_density = FT(chain.bulk_density) / FT(N)
            for s in 1:N
                α = seg_spec[s]
                bulk[α] += chain_density
            end
        end
    end

    for solvent in system.solvents
        if solvent.ensemble == :grand_canonical
            bulk[solvent.species_index] += FT(solvent.bulk_density)
        else
            bulk[solvent.species_index] += FT(solvent.n_molecules) / V_eff
        end
    end

    return bulk
end
