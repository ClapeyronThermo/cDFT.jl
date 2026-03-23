"""
    propagate_scft!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, buf, P, iP)

Run DGC forward/backward propagator sweeps using shifted fields `Δw = w - w_bulk`.

Shifting by `w_bulk` keeps propagator values near O(1) for near-uniform systems,
avoiding the numerical underflow that occurs when raw fields are large.
The shifted propagator satisfies `q̃(r,s) = q(r,s) * exp(Σ_{t=1}^{s} w_bulk[α(t)])`,
so `Q̃ ≈ 1` for uniform systems (instead of `Q ∼ exp(-N * w_bulk) ≈ 0`).
"""
function propagate_scft!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
    nd = dimension(system)
    propagator = system.propagator
    nchains = length(propagator.N)

    for c in 1:nchains
        Nc = propagator.N[c]
        seg_spec = propagator.segment_species[c]

        # Forward propagator with shifted fields
        α1 = seg_spec[1]
        selectdim(q_fwd[c], nd+1, 1) .= exp.(w_bulk[α1] .- selectdim(w, nd+1, α1))

        for i in 2:Nc
            αi = seg_spec[i]
            bond_key = minmax(seg_spec[i-1], seg_spec[i])
            kernel = propagator.kernel_map[bond_key]
            convolve!(selectdim(q_fwd[c], nd+1, i), selectdim(q_fwd[c], nd+1, i-1), kernel, P, iP, buf)
            selectdim(q_fwd[c], nd+1, i) .*= exp.(w_bulk[αi] .- selectdim(w, nd+1, αi))
        end

        # Backward propagator with shifted fields
        αN = seg_spec[Nc]
        selectdim(q_bwd[c], nd+1, 1) .= exp.(w_bulk[αN] .- selectdim(w, nd+1, αN))

        for i in 2:Nc
            αi = seg_spec[Nc - i + 1]
            bond_key = minmax(seg_spec[Nc - i + 1], seg_spec[Nc - i + 2])
            kernel = propagator.kernel_map[bond_key]
            convolve!(selectdim(q_bwd[c], nd+1, i), selectdim(q_bwd[c], nd+1, i-1), kernel, P, iP, buf)
            selectdim(q_bwd[c], nd+1, i) .*= exp.(w_bulk[αi] .- selectdim(w, nd+1, αi))
        end
    end
end

"""
    effective_volume(system::SCFTSystem, dz)

Compute the effective domain volume consistent with the Simpson quadrature rule.
Uses `V_eff = ∫(ones, dz)` so that partition functions and densities are
self-consistent regardless of quadrature accuracy.
"""
function effective_volume(system::SCFTSystem, dz)
    ngrid = system.structure.ngrid
    return ∫(ones(ngrid...), dz)
end

"""
    compute_partition_functions(system::SCFTSystem, w, w_bulk, q_fwd, dz)

Compute single-chain partition functions from shifted propagators:
    Q̃_c = (1/V_eff) ∫ q̃_fwd[c](:, N_c) dr

where `q̃` is the propagator computed with shifted fields `Δw = w - w_bulk`.
For a uniform system at bulk densities, `Q̃ ≈ 1`.

For solvents:
    Q̃_S = (1/V_eff) ∫ exp(w_S^b - w_S(r)) dr

Returns `(Q_chains::Vector{Float64}, Q_solvents::Vector{Float64})`.
"""
function compute_partition_functions(system::SCFTSystem, w, w_bulk, q_fwd, dz)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    propagator = system.propagator
    nchains = length(propagator.N)

    V_eff = effective_volume(system, dz)

    # Chain partition functions (from shifted propagators)
    Q_chains = Vector{Float64}(undef, nchains)
    for c in 1:nchains
        Nc = propagator.N[c]
        q_end = Array(selectdim(q_fwd[c], nd+1, Nc))
        Q_chains[c] = ∫(q_end, dz) / V_eff
    end

    # Solvent partition functions (shifted: exp(w_bulk - w))
    Q_solvents = Vector{Float64}(undef, length(system.solvents))
    for (s, solvent) in enumerate(system.solvents)
        sp = solvent.species_index
        exp_shifted = exp.(w_bulk[sp] .- Array(selectdim(w, nd+1, sp)))
        Q_solvents[s] = ∫(exp_shifted, dz) / V_eff
    end

    return Q_chains, Q_solvents
end

"""
    compute_densities!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ)

Compute density profiles from shifted propagators and partition functions.

For chain segments with shifted propagators:
    ρ_α(r) += prefactor * Σ_{s: α(s)=α} q̃_fwd(r,s) * q̃_bwd(r, N+1-s) * exp(Δw_α(r))

where `Δw = w - w_bulk`, and the exp(Δw) corrects for double-counting of the
Boltzmann weight at segment s. The shift factors cancel between numerator and
denominator (Q̃), keeping values near O(1).

For solvents (grand canonical):
    ρ_α(r) = ρ_S^b * exp(w_S^b - w_S(r))

For solvents (canonical):
    ρ_α(r) = (n_S / V_eff) * exp(w_S^b - w_S(r)) / Q̃_S
"""
function compute_densities!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    propagator = system.propagator
    nchains = length(propagator.N)
    nspecies = system.nspecies
    dz = structure_dz(system.structure)

    V_eff = effective_volume(system, dz)

    # Zero out densities
    ρ .= 0.0

    # Chain contributions
    for c in 1:nchains
        chain = system.chains[c]
        Nc = propagator.N[c]
        seg_spec = propagator.segment_species[c]
        Qc = Q_chains[c]

        # Prefactor depends on ensemble
        if chain.ensemble == :canonical
            prefactor = chain.n_chains / (V_eff * Qc)
        else
            prefactor = chain.bulk_density / (chain.N * Qc)
        end

        # For each segment, add contribution to the appropriate species
        # With shifted propagators, the double-count correction uses Δw = w - w_bulk
        for s in 1:Nc
            α = seg_spec[s]
            selectdim(ρ, nd+1, α) .+= prefactor .* selectdim(q_fwd[c], nd+1, s) .*
                selectdim(q_bwd[c], nd+1, Nc + 1 - s) .*
                exp.(selectdim(w, nd+1, α) .- w_bulk[α])
        end
    end

    # Solvent contributions
    for (s, solvent) in enumerate(system.solvents)
        sp = solvent.species_index
        if solvent.ensemble == :grand_canonical
            # ρ_S(r) = ρ_S^b * exp(w_S^b - w_S(r))
            selectdim(ρ, nd+1, sp) .+= solvent.bulk_density .* exp.(w_bulk[sp] .- selectdim(w, nd+1, sp))
        else
            # Canonical solvent: ρ_S = (n_S/V_eff) * exp(w_S^b - w_S(r)) / Q̃_S
            Qs = Q_solvents[s]
            selectdim(ρ, nd+1, sp) .+= (solvent.n_molecules / V_eff) .*
                exp.(w_bulk[sp] .- selectdim(w, nd+1, sp)) ./ Qs
        end
    end
end
