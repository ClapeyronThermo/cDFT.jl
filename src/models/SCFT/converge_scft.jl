"""
    scft_iterate!(system::SCFTSystem, ρ; kwargs...)

SCFT iteration loop: Picard mixing followed by Anderson acceleration.

Protocol:
1. Run Picard (w += β·r) while err ≥ `anderson_start`, accumulating field/
   residual history in a circular buffer on every iteration.
2. Switch to Anderson acceleration once err < `anderson_start`, using the
   already-accumulated history immediately (no history loss at the switch).

The Anderson update (Walker & Ni, 2011, Eq. 2.5) at iteration k:
   ΔF_i = r_{k-i+1} - r_{k-i},   Δx_i = w_{k-i+1} - w_{k-i}
   solve:  [ΔFᵢ·ΔFⱼ] γ = [ΔFᵢ·rₖ]   (m_eff × m_eff least-squares)
   update: w ← w + β·r - Σᵢ γᵢ·(Δxᵢ + β·ΔFᵢ)

Stability condition (Picard): `beta < 2 / (1 + kappa)`.
For kappa=20 the default beta=0.01 gives ~10× safety margin.

# Keyword Arguments
- `maxit::Int=5000`: Maximum total iterations.
- `beta::Float64=0.01`: Mixing coefficient for both Picard and Anderson steps.
- `tol::Float64=1e-6`: Convergence tolerance on max|r|.
- `anderson_start::Float64=1e-2`: Switch from Picard to Anderson when err < this.
- `anderson_m::Int=5`: Anderson history length (number of past iterates kept).
- `quadrature::Symbol=:trapz`: Quadrature rule for partition-function integrals.
  `:trapz` — periodic composite trapezoidal (uniform weights, spectral convergence for
  smooth periodic functions, works for any N). Recommended for periodic SCFT domains.
  `:simpson` — composite Simpson rule (O(h⁴), requires odd N per dimension).
- `log_interval::Int=100`: Log every N iterations (0 = never).
- `save_interval::Int=0`: Call `save_callback` every N iterations (0 = never).
- `save_callback`: `f(iter, ρ_array, w_array)` called at each save interval.
- `verbose::Bool=true`: Print convergence summary and solver-switch message.

# Returns
Named tuple `(converged, iter, error)`.
"""
function scft_iterate!(system::SCFTSystem, ρ;
    maxit          :: Int     = 5000,
    beta                     = 0.01,
    tol                      = 1e-6,
    anderson_start           = 1e-2,
    anderson_m     :: Int     = 5,
    quadrature     :: Symbol  = :trapz,
    log_interval   :: Int     = 100,
    save_interval  :: Int     = 0,
    save_callback             = nothing,
    verbose        :: Bool    = true,
)
    device   = system.options.device
    dz       = structure_dz(system.structure)
    nd       = dimension(system)
    nspecies = system.nspecies
    FT       = eltype(ρ)
    beta_ft  = FT(beta)
    tol_ft   = FT(tol)
    anderson_start_ft = FT(anderson_start)

    w     = similar(ρ)
    w_new = similar(ρ)
    r     = similar(ρ)   # residual: r_k = w_new(w_k) - w_k

    q_fwd, q_bwd, buf_r, buf_c, P, iP = preallocate_propagator(system, system.propagator, ρ, device)
    batched_fft_cache = preallocate_scft_batched_fft_cache(system, ρ)

    # Precompute quadrature weights on the device (shape = ngrid).
    # :trapz — uniform weights prod(dz), spectral convergence for periodic domains.
    # :simpson — composite Simpson 1/4/2 pattern, O(h⁴), requires odd N.
    weights_gpu = if quadrature == :trapz
        trapz_weights(system.structure.ngrid, dz, system.options)
    elseif quadrature == :simpson
        simpson_weights(system.structure.ngrid, dz, system.options)
    else
        error("Unknown quadrature rule: $quadrature. Use :trapz or :simpson.")
    end

    # V_eff is constant — compute once from the weight array.
    V_eff = sum(weights_gpu)

    # Print the effective volume
    @info "Effective volume: $V_eff"

    # Compute bulk densities using the same V_eff as the iteration, so canonical
    # prefactors (n_chains / V_eff) are consistent with the chosen quadrature rule.
    bulk   = compute_bulk_densities(system; V_eff=V_eff)
    w_bulk = compute_bulk_fields(system.interaction, bulk)

    # Preallocate scratch buffer for compute_fields! to avoid per-call GPU allocations.
    scratch = similar(selectdim(w, nd+1, 1))

    # Preallocate exp_field[α] = exp(w_bulk[α] - w_α) and its inverse,
    # one array per species.  Reused every iteration to avoid redundant GPU kernels
    # when multiple segments share the same species (common in diblock chains).
    exp_field     = [similar(selectdim(w, nd+1, 1)) for _ in 1:nspecies]
    inv_exp_field = [similar(selectdim(w, nd+1, 1)) for _ in 1:nspecies]

    compute_fields!(system, ρ, w; scratch=scratch)

    # ── Anderson history (circular buffer, m+1 slots for m differences) ────────
    m      = anderson_m
    aa_w   = [similar(w) for _ in 1:m+1]   # field values
    aa_r   = [similar(w) for _ in 1:m+1]   # residuals
    ΔF_buf = [similar(w) for _ in 1:m]     # preallocated ΔF work arrays
    Δx_buf = [similar(w) for _ in 1:m]     # preallocated Δx work arrays
    aa_count       = 0      # total history entries stored so far
    using_anderson = false

    # GEMM scratch for Anderson — on-device column matrix, avoids scalar dot sync points.
    # ΔF_mat has shape (N_flat, m) where N_flat = total field elements (can be millions).
    # GPU does the heavy (N_flat) computation; only the tiny (m×m) result transfers to CPU.
    N_flat = length(w)
    ΔF_mat = similar(w, N_flat, m)

    err = FT(Inf)

    for iter in 1:maxit

        # ── Precompute exp_field[α] = exp(w_bulk[α] - w_α) once per iteration.
        # This is reused by propagate_scft!, compute_partition_functions, and
        # compute_densities! instead of recomputing per segment or per call.
        # Note: selectdim must be called outside @. to get the view first.
        for α in 1:nspecies
            w_α = selectdim(w, nd+1, α)
            @. exp_field[α]     = exp(w_bulk[α] - w_α)
            @. inv_exp_field[α] = one(FT) / exp_field[α]
        end

        # ── Propagate, compute densities and new fields ───────────────────────
        propagate_scft!(system, w, w_bulk, q_fwd, q_bwd, buf_r, buf_c, P, iP;
                        exp_field=exp_field, batched_fft_cache=batched_fft_cache)
        Q_chains, Q_solvents = compute_partition_functions(system, w, w_bulk, q_fwd, dz;
                                   weights=weights_gpu, V_eff=V_eff, exp_field=exp_field)
        compute_densities!(system, w, w_bulk, q_fwd, q_bwd, Q_chains, Q_solvents, ρ;
                           V_eff=V_eff, exp_field=exp_field, inv_exp_field=inv_exp_field)
        compute_fields!(system, ρ, w_new; scratch=scratch)

        # ── Residual and error ────────────────────────────────────────────────
        @. r = w_new - w
        err  = maximum(abs, Array(r))

        # ── Logging ───────────────────────────────────────────────────────────
        if log_interval > 0 && iter % log_interval == 0
            H     = free_energy(system, ρ, w, Q_chains, Q_solvents; V_eff=V_eff, w_bulk=w_bulk)
            phase = using_anderson ? "AA" : "Picard"
            @info "SCFT iter $(lpad(iter, 5)) | err = $(round(err; sigdigits=3)) | F = $(round(H; sigdigits=6)) | $(phase)"
        end

        if save_interval > 0 && iter % save_interval == 0 && save_callback !== nothing
            save_callback(iter, Array(ρ), Array(w))
        end

        # ── Convergence check ─────────────────────────────────────────────────
        if err < tol_ft
            if verbose
                H = free_energy(system, ρ, w, Q_chains, Q_solvents; V_eff=V_eff, w_bulk=w_bulk)
                @info "SCFT converged at iter $(iter): err = $(round(err; sigdigits=3)) | F = $(round(H; sigdigits=6))"
            end
            return (converged=true, iter=iter, error=err)
        end

        # ── Store (w, r) in history on EVERY iteration (Picard or Anderson) ──
        # This ensures history is ready when the solver switches.
        slot = mod1(aa_count + 1, m + 1)
        aa_w[slot] .= w
        aa_r[slot] .= r
        aa_count += 1

        # ── Decide whether to switch to Anderson ──────────────────────────────
        if !using_anderson && err < anderson_start_ft && aa_count >= 2
            using_anderson = true
            if verbose
                @info "SCFT switching to Anderson acceleration at iter $(iter) (err = $(round(err; sigdigits=3)), m=$(m))"
            end
        end

        # ── Field update ──────────────────────────────────────────────────────
        if using_anderson && aa_count >= 2

            m_eff = min(aa_count - 1, m)   # number of difference vectors available

            # Fill preallocated difference buffers (newest difference first)
            for i in 1:m_eff
                s_new = mod1(aa_count - i + 1, m + 1)
                s_old = mod1(aa_count - i,     m + 1)
                @. ΔF_buf[i] = aa_r[s_new] - aa_r[s_old]
                @. Δx_buf[i] = aa_w[s_new] - aa_w[s_old]
            end

            # Pack ΔF columns into device matrix for GEMM.
            # vec() is a zero-copy reshape since ΔF_buf[i] is contiguous via similar(w).
            for i in 1:m_eff
                ΔF_mat[:, i] .= vec(ΔF_buf[i])
            end

            # Build (m_eff × m_eff) Gram matrix and rhs via a single GEMM + GEMV on device.
            # GPU sums over N_flat (millions) of rows; only the tiny (m_eff×m_eff) and
            # (m_eff,) results are transferred to CPU. Array() is a no-op on CPU arrays.
            DF = view(ΔF_mat, :, 1:m_eff)
            G  = Array(DF' * DF)        # GEMM: (m_eff × N_flat) * (N_flat × m_eff) → (m_eff × m_eff)
            g  = Array(DF' * vec(r))    # GEMV: (m_eff × N_flat) * (N_flat,)         → (m_eff,)

            # Tikhonov regularization to handle near-singular Gram matrices
            λ = max(1e-12 * tr(G), 1e-16)
            for i in 1:m_eff
                G[i, i] += λ
            end

            γ = try
                FT.(G \ g)
            catch
                # Fall back to a pure Picard step if the solve fails
                zeros(FT, m_eff)
            end

            # Anderson update: w ← w + β·r - Σᵢ γᵢ·(Δxᵢ + β·ΔFᵢ)
            w .+= beta_ft .* r
            for i in 1:m_eff
                @. w -= γ[i] * (Δx_buf[i] + beta_ft * ΔF_buf[i])
            end

        else
            # Pure Picard: w += β · r
            w .+= beta_ft .* r
        end

    end  # iter loop

    if verbose
        @warn "SCFT did not converge after $(maxit) iterations (err = $(round(err; sigdigits=3)))"
    end
    return (converged=false, iter=maxit, error=err)
end


"""
    converge_fields!(system::SCFTSystem, ρ; kwargs...)

Field-based SCFT convergence using Picard mixing followed by Anderson
acceleration. Updates `ρ` in-place.

Key parameters (forwarded to `scft_iterate!`):
- `maxit=5000`, `beta=0.01`, `tol=1e-6`
- `anderson_start=1e-2`: switch from Picard to Anderson when err < this
- `anderson_m=5`: number of Anderson history vectors
- `quadrature=:trapz`: `:trapz` (periodic trapezoidal, default) or `:simpson`
- `log_interval=100`, `verbose=true`
- `save_interval=0`, `save_callback=nothing`

Returns `(converged, iter, error)`.
"""
function converge_fields!(system::SCFTSystem, ρ; kwargs...)
    return scft_iterate!(system, ρ; kwargs...)
end

"""
    converge!(system::SCFTSystem, ρ; kwargs...)

Alias for `converge_fields!`. Keyword arguments forwarded to `scft_iterate!`.
"""
function converge!(system::SCFTSystem, ρ; kwargs...)
    return scft_iterate!(system, ρ; kwargs...)
end
