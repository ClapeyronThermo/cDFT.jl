"""
    converge!(system::SCFTSystem, ρ; picard_maxit=2000, anderson_maxit=10000, beta=1e-2, rtol=1e-8, atol=1e-8)

Density-based SCFT convergence. Iterates on density profiles using Anderson
mixing (via SIAMFANLEquations.aasol).

The fixed-point equation is:
    ρ → fields(ρ) → propagators(fields) → Q(propagators) → ρ_new(propagators, Q)
    G(ρ) = ρ_new - ρ = 0
"""
function converge!(system::SCFTSystem, ρ; picard_maxit=2000, anderson_maxit=10000, beta=1e-2, rtol=1e-8, atol=1e-8)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    nspecies = system.nspecies
    device = system.options.device
    propagator = system.propagator

    dz = structure_dz(system.structure)

    # Preallocate working arrays
    w = similar(ρ)
    ρ_new = similar(ρ)

    # Propagator cache
    nchains = length(propagator.N)
    q_fwd, q_bwd, buf, P, iP = preallocate_propagator(
        system, propagator, ρ, device
    )

    # Compute bulk densities and fields for normalization
    bulk = compute_bulk_densities(system)
    w_bulk = compute_bulk_fields(system.interaction, bulk)

    function obj!(G, x)
        ρ .= Adapt.adapt(device, reshape(x, size(ρ)))
        clamp!(ρ, 1e-15, Inf)

        compute_fields!(system, ρ, w)
        propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
        Q_chains, Q_solvents = compute_partition_functions(system, w, w_bulk, q_fwd, dz)
        compute_densities!(system, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ_new)

        G .= vec(Adapt.adapt(CPU(), ρ_new .- ρ))
        return G
    end

    x0 = Adapt.adapt(CPU(), vec(copy(ρ)))
    n = length(x0)

    # Phase 1: Picard iterations (m=0, skip if picard_maxit == 0)
    if picard_maxit > 0
        result = SIAMFANLEquations.aasol(obj!, x0, 0, zeros(n, 4);
            beta=beta, rtol=1e-1, atol=1e-1, maxit=picard_maxit)
        x_start = result.solution
    else
        x_start = x0
    end

    # Phase 2: Anderson mixing
    result = SIAMFANLEquations.aasol(obj!, x_start, 5, zeros(n, 14);
        beta=beta, rtol=rtol, atol=atol, maxit=anderson_maxit)

    ρ .= Adapt.adapt(device, reshape(result.solution, size(ρ)))
    clamp!(ρ, 1e-15, Inf)

    return result
end

"""
    converge_fields!(system::SCFTSystem, ρ; picard_maxit=2000, anderson_maxit=10000, beta=1e-2, rtol=1e-8, atol=1e-8)

Field-based SCFT convergence. Iterates on field profiles using Anderson mixing.

The fixed-point equation is:
    w → propagators(w) → Q → ρ(propagators, Q) → w_new(ρ)
    G(w) = w_new - w = 0

Returns the converged fields in `w` and updates `ρ` to the final densities.
"""
function converge_fields!(system::SCFTSystem, ρ; picard_maxit=2000, anderson_maxit=10000, beta=1e-2, rtol=1e-8, atol=1e-8)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    nspecies = system.nspecies
    device = system.options.device
    propagator = system.propagator

    dz = structure_dz(system.structure)

    # Preallocate
    w = similar(ρ)
    w_new = similar(ρ)

    nchains = length(propagator.N)
    q_fwd, q_bwd, buf, P, iP = preallocate_propagator(
        system, propagator, ρ, device
    )

    bulk = compute_bulk_densities(system)
    w_bulk = compute_bulk_fields(system.interaction, bulk)

    # Initialize fields from initial density
    compute_fields!(system, ρ, w)

    function obj!(G, x)
        w .= Adapt.adapt(device, reshape(x, size(w)))

        propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
        Q_chains, Q_solvents = compute_partition_functions(system, w, w_bulk, q_fwd, dz)
        compute_densities!(system, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ)
        clamp!(ρ, 1e-15, Inf)
        compute_fields!(system, ρ, w_new)

        G .= vec(Adapt.adapt(CPU(), w_new .- w))
        return G
    end

    x0 = Adapt.adapt(CPU(), vec(copy(w)))
    n = length(x0)

    # Phase 1: Picard (skip if picard_maxit == 0)
    if picard_maxit > 0
        result = SIAMFANLEquations.aasol(obj!, x0, 0, zeros(n, 4);
            beta=beta, rtol=1e-1, atol=1e-1, maxit=picard_maxit)
        x_start = result.solution
    else
        x_start = x0
    end

    # Phase 2: Anderson
    result = SIAMFANLEquations.aasol(obj!, x_start, 5, zeros(n, 14);
        beta=beta, rtol=rtol, atol=atol, maxit=anderson_maxit)

    # Final density from converged fields
    w .= Adapt.adapt(device, reshape(result.solution, size(w)))
    propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf, P, iP)
    Q_chains, Q_solvents = compute_partition_functions(system, w, w_bulk, q_fwd, dz)
    compute_densities!(system, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ)
    clamp!(ρ, 1e-15, Inf)

    return result
end
