"""
    propagate_scft!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, buf_r, buf_c, P, iP)

Run DGC forward/backward propagator sweeps using shifted fields `Δw = w - w_bulk`.

Shifting by `w_bulk` keeps propagator values near O(1) for near-uniform systems,
avoiding the numerical underflow that occurs when raw fields are large.
The shifted propagator satisfies `q̃(r,s) = q(r,s) * exp(Σ_{t=1}^{s} w_bulk[α(t)])`,
so `Q̃ ≈ 1` for uniform systems (instead of `Q ∼ exp(-N * w_bulk) ≈ 0`).
"""
function propagate_scft!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, buf_r, buf_c, P, iP;
                         exp_field=nothing)
    nd = dimension(system)
    propagator = system.propagator
    nchains = length(propagator.N)

    # Helper: return precomputed exp_field[α] if available, else compute on the fly.
    # exp_field[α] = exp(w_bulk[α] - w_α(r))
    ef(α) = exp_field !== nothing ? exp_field[α] :
                exp.(w_bulk[α] .- selectdim(w, nd+1, α))

    for c in 1:nchains
        Nc = propagator.N[c]
        seg_spec = propagator.segment_species[c]

        # Forward propagator with shifted fields
        α1 = seg_spec[1]
        selectdim(q_fwd[c], nd+1, 1) .= ef(α1)

        for i in 2:Nc
            αi = seg_spec[i]
            bond_key = minmax(seg_spec[i-1], seg_spec[i])
            kernel = propagator.kernel_map[bond_key]
            convolve!(selectdim(q_fwd[c], nd+1, i), selectdim(q_fwd[c], nd+1, i-1), kernel, P, iP, buf_r, buf_c)
            selectdim(q_fwd[c], nd+1, i) .*= ef(αi)
        end

        # Backward propagator with shifted fields
        αN = seg_spec[Nc]
        selectdim(q_bwd[c], nd+1, 1) .= ef(αN)

        for i in 2:Nc
            αi = seg_spec[Nc - i + 1]
            bond_key = minmax(seg_spec[Nc - i + 1], seg_spec[Nc - i + 2])
            kernel = propagator.kernel_map[bond_key]
            convolve!(selectdim(q_bwd[c], nd+1, i), selectdim(q_bwd[c], nd+1, i-1), kernel, P, iP, buf_r, buf_c)
            selectdim(q_bwd[c], nd+1, i) .*= ef(αi)
        end
    end
end

"""
    effective_volume(system::SCFTSystem, dz)

Compute the effective domain volume using the periodic trapezoidal rule:
    V_eff = prod(dz) * prod(ngrid) = prod(L_i)

This is exact for any N and consistent with the default `:trapz` quadrature used
in `scft_iterate!`. The previous Simpson-based fallback gave ~5% error for typical
3D grids because `structure_dz` returns `L/N` (periodic spacing) rather than
`L/(N-1)` (non-periodic), causing the Simpson weights to underestimate the volume.
"""
function effective_volume(system::SCFTSystem, dz)
    ngrid = system.structure.ngrid
    return prod(dz) * prod(ngrid)
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
function compute_partition_functions(system::SCFTSystem, w, w_bulk, q_fwd, dz;
                                     weights=nothing, V_eff=nothing, exp_field=nothing)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    propagator = system.propagator
    nchains = length(propagator.N)

    V_eff = V_eff !== nothing ? V_eff : effective_volume(system, dz)

    # Chain partition functions (from shifted propagators)
    Q_chains = Vector{Float64}(undef, nchains)
    for c in 1:nchains
        Nc = propagator.N[c]
        if weights !== nothing
            # GPU-friendly: dot product with precomputed weight array — no host transfer
            Q_chains[c] = sum(selectdim(q_fwd[c], nd+1, Nc) .* weights) / V_eff
        else
            q_end = Array(selectdim(q_fwd[c], nd+1, Nc))
            Q_chains[c] = ∫(q_end, dz) / V_eff
        end
    end

    # Solvent partition functions (shifted: exp(w_bulk - w))
    Q_solvents = Vector{Float64}(undef, length(system.solvents))
    for (s, solvent) in enumerate(system.solvents)
        sp = solvent.species_index
        # Use precomputed exp_field[sp] if available (avoids recomputing exp per call)
        ef_sp = exp_field !== nothing ? exp_field[sp] :
                    exp.(w_bulk[sp] .- selectdim(w, nd+1, sp))
        if weights !== nothing
            Q_solvents[s] = sum(ef_sp .* weights) / V_eff
        else
            exp_shifted = exp_field !== nothing ? Array(ef_sp) :
                              exp.(w_bulk[sp] .- Array(selectdim(w, nd+1, sp)))
            Q_solvents[s] = ∫(exp_shifted, dz) / V_eff
        end
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
function compute_densities!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ;
                            V_eff=nothing, exp_field=nothing, inv_exp_field=nothing)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    propagator = system.propagator
    nchains = length(propagator.N)
    nspecies = system.nspecies
    dz = structure_dz(system.structure)

    V_eff = V_eff !== nothing ? V_eff : effective_volume(system, dz)

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

        # For each segment, add contribution to the appropriate species.
        # Double-count correction: exp(w_α - w_bulk_α) = 1/exp_field[α].
        # Use precomputed inv_exp_field if available to avoid recomputing per segment.
        for s in 1:Nc
            α = seg_spec[s]
            inv_ef_α = inv_exp_field !== nothing ? inv_exp_field[α] :
                           exp.(selectdim(w, nd+1, α) .- w_bulk[α])
            selectdim(ρ, nd+1, α) .+= prefactor .* selectdim(q_fwd[c], nd+1, s) .*
                selectdim(q_bwd[c], nd+1, Nc + 1 - s) .* inv_ef_α
        end
    end

    # Solvent contributions
    for (s, solvent) in enumerate(system.solvents)
        sp = solvent.species_index
        # Use precomputed exp_field[sp] if available
        ef_sp = exp_field !== nothing ? exp_field[sp] :
                    exp.(w_bulk[sp] .- selectdim(w, nd+1, sp))
        if solvent.ensemble == :grand_canonical
            # ρ_S(r) = ρ_S^b * exp(w_S^b - w_S(r))
            selectdim(ρ, nd+1, sp) .+= solvent.bulk_density .* ef_sp
        else
            # Canonical solvent: ρ_S = (n_S/V_eff) * exp(w_S^b - w_S(r)) / Q̃_S
            Qs = Q_solvents[s]
            selectdim(ρ, nd+1, sp) .+= (solvent.n_molecules / V_eff) .* ef_sp ./ Qs
        end
    end
end
