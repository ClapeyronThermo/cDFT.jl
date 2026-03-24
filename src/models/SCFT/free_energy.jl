"""
    free_energy(system::SCFTSystem, ρ, w, Q_chains, Q_solvents)

Compute the SCFT free energy (mean-field Hamiltonian):

    H = U_int + U_comp - Σ_K ∫w_K ρ_K dr - Σ_c n_c ln(Q_c) - z_S V Q_S

`Q_chains` and `Q_solvents` are the *shifted* partition functions (Q̃) from
propagators computed with `Δw = w - w_bulk`. The relationship to the true
partition function is `Q = Q̃ * exp(-Σ_s w_bulk[α(s)])`, and the free energy
terms are adjusted accordingly.
"""
function free_energy(system::SCFTSystem, ρ, w, Q_chains, Q_solvents;
                     V_eff=nothing, w_bulk=nothing)
    free_energy(system.interaction, system, ρ, w, Q_chains, Q_solvents;
                V_eff=V_eff, w_bulk=w_bulk)
end

function free_energy(fh::FloryHuggins, system::SCFTSystem, ρ, w, Q_chains, Q_solvents;
                     V_eff=nothing, w_bulk=nothing)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    nspecies = system.nspecies
    dz = structure_dz(system.structure)

    # Use V_eff and w_bulk from the iteration loop when provided, so the free energy
    # is computed consistently with the quadrature rule chosen for the iteration.
    # Fallback to recomputing from scratch (CPU Simpson) for standalone calls.
    if V_eff === nothing
        V_eff = effective_volume(system, dz)
    end
    if w_bulk === nothing
        bulk = compute_bulk_densities(system; V_eff=V_eff)
        w_bulk = compute_bulk_fields(fh, bulk)
    end

    chi = fh.chi
    rho0 = fh.rho0
    kappa = fh.kappa

    # U_int = (1/ρ₀) ∫ Σ_{α<β} χ_αβ ρ_α ρ_β dr
    U_int_integrand = zeros(ngrid...)
    for α in 1:nspecies
        for β in (α+1):nspecies
            if chi[α, β] != 0.0
                U_int_integrand .+= chi[α, β] .* Array(selectdim(ρ, nd+1, α)) .* Array(selectdim(ρ, nd+1, β))
            end
        end
    end
    U_int = ∫(U_int_integrand, dz) / rho0

    # U_comp = (ζ / 2) ∫ (ρ₊/ρ₀ - 1)² dr
    # Consistent with w_comp = ζ/ρ₀ · (ρ₊/ρ₀ − 1) = δU_comp/δρ_α.
    ρ_total = zeros(ngrid...)
    for α in 1:nspecies
        ρ_total .+= Array(selectdim(ρ, nd+1, α))
    end
    U_comp_integrand = (kappa / 2.0) .* (ρ_total ./ rho0 .- 1.0) .^ 2
    U_comp = ∫(U_comp_integrand, dz)

    # -Σ_K ∫ w_K ρ_K dr
    wρ_sum = 0.0
    for α in 1:nspecies
        wρ_integrand = Array(selectdim(w, nd+1, α)) .* Array(selectdim(ρ, nd+1, α))
        wρ_sum += ∫(wρ_integrand, dz)
    end

    # Chain contributions: -n_c * ln(Q_c) where Q_c is the TRUE partition function
    # Q̃_c = Q_c * exp(Σ_s w_bulk[α(s)]), so ln(Q_c) = ln(Q̃_c) - Σ_s w_bulk[α(s)]
    chain_sum = 0.0
    for (c, chain) in enumerate(system.chains)
        seg_spec = system.propagator.segment_species[c]
        w_bulk_sum = sum(w_bulk[seg_spec[s]] for s in 1:chain.N)
        ln_Qc_true = log(Q_chains[c]) - w_bulk_sum

        if chain.ensemble == :canonical
            chain_sum -= chain.n_chains * ln_Qc_true
        else
            chain_sum -= (chain.bulk_density / chain.N) * V_eff * ln_Qc_true
        end
    end

    # Solvent contributions
    # For solvents: Q̃_S = Q_S * exp(w_bulk[sp]), so Q_S = Q̃_S * exp(-w_bulk[sp])
    # Grand canonical: -z_S * V * Q_S = -ρ_S^b * exp(w_S^b) * V * Q̃_S * exp(-w_S^b) = -ρ_S^b * V * Q̃_S
    solvent_sum = 0.0
    for (s, solvent) in enumerate(system.solvents)
        sp = solvent.species_index
        if solvent.ensemble == :grand_canonical
            solvent_sum -= solvent.bulk_density * V_eff * Q_solvents[s]
        else
            ln_Qs_true = log(Q_solvents[s]) - w_bulk[sp]
            solvent_sum -= solvent.n_molecules * ln_Qs_true
        end
    end

    H = U_int + U_comp - wρ_sum + chain_sum + solvent_sum
    return H
end
