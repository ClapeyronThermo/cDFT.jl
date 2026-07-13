#wrapper for the AASol constructor that uses the keyword arguments in converge!

function __AASol(;maxit, beta, tol, anderson_start, anderson_m, verbose)
    return AASol(anderson_m;
            beta=beta, rtol=tol, atol=tol, maxit=maxit,
            picard_maxit=maxit, picard_beta=beta,
            picard_rtol=anderson_start, picard_atol=anderson_start,
            verbose=verbose)
end

#dispatches for specific defaults
__AASol(system::Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem};maxit=10000,beta=1e-2,tol= 1e-4,anderson_start=1e-1,anderson_m=5,verbose=false) = __AASol(;maxit, beta, tol, anderson_start, anderson_m, verbose)
__AASol(system::SCFTSystem;maxit=5000,beta=1e-1,tol= 1e-6,anderson_start=1e-2,anderson_m=5,verbose=false) = __AASol(;maxit, beta, tol, anderson_start, anderson_m, verbose)

struct cDFTProfileSolver{A,C}
    fixpoint::A
    quadrature:: Symbol
    log_interval::Int
    save_interval::Int
    save_callback::C
    verbose:: Bool
end

function cDFTProfileSolver(fixpoint_method::T;
                            quadrature              = :trapz,
                            log_interval            = 100,
                            save_interval           = 0,
                            save_callback           = nothing,
                            verbose                 = false,) where T


    cDFTProfileSolver(fixpoint_method,quadrature,log_interval,save_interval,save_callback,verbose)
end

"""
    get_new_profile!(system, ρ, δfδρ_res, caches)

One DFT-family fixed-point step: given the current density profile `ρ`, evaluate the
functional derivative `δfδρ_res` (bulk residual `δFδρ_res!` + external field
`evaluate_external_field!` + propagator chain contributions `propagate!`), then
assemble the new log-density-space field guess into `caches.ln_Gx` from the per-species
chemical-potential balance (plus the electrostatic constant-potential correction for
electrolyte systems). Mirrors `get_new_profile!(system::SCFTSystem, ρ, w, caches)`
(below, in this same file), which bundles SCFT's per-iteration
propagate→partition-functions→densities→fields sequence into the same
`(system, ρ, field, caches)` shape.

`caches` is a named tuple of buffers preallocated once outside the iteration loop:
`cache_model, cache_external, cache_propagator, ln_Gx`.
"""
function get_new_profile!(system::Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem}, ρ, δfδρ_res, caches)
    (; cache_model, cache_external, cache_propagator, ln_Gx) = caches
    FP = fptype(system.options)
    nd = dimension(system)
    species = system.species
    model = system.model

    δFδρ_res!(system, ρ, δfδρ_res, cache_model...)

    evaluate_external_field!(system, ρ, δfδρ_res, cache_external)

    propagate!(system, ρ, δfδρ_res, cache_propagator)

    chem_pot_res_dens = FP.(species.chempot_res .+ log.(species.bulk_density))

    for i in @comps
        chem_pot_res_dens_i = chem_pot_res_dens[i]
        for k in @chain(i)
            if system.species.nbeads[i] != 1
                α = findall(model.groups.n_intergroups[i][k,:] .== 1 .&& species.levels .> species.levels[k])
            else
                α = k
            end

            # All operations stay vectorized on GPU
            selectdim(ln_Gx, nd+1, k) .=  chem_pot_res_dens_i .-
                                        selectdim(δfδρ_res, nd+1, k)
        end
    end

    if any(typeof.(system.external_field) .<: ElectrostaticPotentialModel)
        ep_model = filter(x -> x isa ElectrostaticPotentialModel, system.external_field)[1]
        Z = model.charge

        psi_c = find_ψ_const(system.structure, ep_model, model, exp.(ln_Gx))/k_B/system.structure.conditions[2]
        for i in @comps
            for k in @chain(i)
                selectdim(ln_Gx,nd+1,k) .-= psi_c*Z[k]
            end
        end
    end

    clamp!(ln_Gx, -100, 100)

    return nothing
end

"""
    converge!(system::DFTSystem, ρ; kwargs...)

For a given system, converge the profiles using the solver specified under `system.options.solver`. Convergence is achieved by solving the generic equation:

```julia
ρi = ρi_bulk*exp(β(μi_res - δFδρ_res))
```

For stability purposes, the equation has be reformulated as:

```julia
ln(ρi) = ln(ρi_bulk) + β(μi_res - δFδρ_res)
```

via `aasol` (`src/utils/anderson.jl`), with an optional Picard warmup phase before
handing off to Anderson mixing. `get_new_profile!` (above) is the fixed-point map
`ln(ρ) -> ln_G(δfδρ_res(ρ))`.

See `converge!(system::SCFTSystem, ρ; kwargs...)` (below, in this same file) for
the SCFT-family method — same function name and kwarg vocabulary, dispatched separately
since SCFT iterates on the field `w` rather than `ln(ρ)`.

# Keyword Arguments
- `maxit::Int=10000`: Maximum Anderson-phase iterations (also used as the Picard-phase cap, since the Picard phase is expected to exit via `anderson_start` well before either cap in practice).
- `beta=1e-3`: Mixing coefficient for both the Picard warmup and Anderson phases.
- `tol=1e-4`: Convergence tolerance (used as both `aasol`'s `rtol` and `atol`).
- `anderson_start=1e-1`: Switch from Picard to Anderson once the residual norm drops below this (used as both `aasol`'s `picard_rtol` and `picard_atol`).
- `anderson_m::Int=0`: Anderson history length (number of past iterates kept). The default of `0` recovers pure (damped) Picard iteration throughout — the historical default behavior. Set to a positive integer to enable real Anderson acceleration.
- `log_interval::Int=0`: Log the free energy every N iterations (0 = never).
- `save_interval::Int=0`: Call `save_callback` every N iterations (0 = never).
- `save_callback`: `f(iter, ρ_array)` called at each save interval.
- `verbose::Bool=false`: Print `aasol`'s per-iteration convergence summary, plus a final convergence/free-energy summary line.
"""
function converge!(system::Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem}, ρ;
    maxit          :: Int  = 10000,
    beta                   = 1e-2,
    tol                    = 1e-4,
    anderson_start         = 1e-1,
    anderson_m     :: Int  = 5,
    log_interval   :: Int  = 0,
    save_interval  :: Int  = 0,
    save_callback           = nothing,
    verbose        :: Bool = false,
)

    FP = fptype(system.options)
    beta = FP(beta)
    tol = FP(tol)
    anderson_start = FP(anderson_start)
    ngrid = system.structure.ngrid
    nd = dimension(system)
    nbeads = size(ρ,nd+1)

    δfδρ_res, cache_model, cache_external, cache_propagator = cDFT.preallocate(system, ρ)
    ln_Gx = similar(ρ)
    caches = (; cache_model, cache_external, cache_propagator, ln_Gx)

    iter_count = Ref(0)

    # aasol's fixed-point map: given the current log-density (flattened), advance one
    # DFT step and return the new log-density guess (flattened).
    function GFix!(ln_G, ln_x)
        ρ .= exp.(reshape(ln_x, (ngrid..., nbeads)))

        get_new_profile!(system, ρ, δfδρ_res, caches)
        iter_count[] += 1

        if log_interval > 0 && iter_count[] % log_interval == 0
            H = free_energy(system, ρ)
            @info "DFT iter $(lpad(iter_count[], 5)) | F = $(round(H; sigdigits=6))"
        end
        if save_interval > 0 && iter_count[] % save_interval == 0 && save_callback !== nothing
            save_callback(iter_count[], Array(ρ))
        end

        copyto!(ln_G, vec(ln_Gx))
        return ln_G
    end

    ln_X0 = vec(log.(ρ))

    m = anderson_m
    Vstore_width = m == 0 ? 4 : 3*m + 3
    result = aasol(GFix!, ln_X0, m, similar(ln_X0, length(ln_X0), Vstore_width);
                 beta=beta, rtol=tol, atol=tol, maxit=maxit,
                 picard_maxit=maxit, picard_beta=beta,
                 picard_rtol=anderson_start, picard_atol=anderson_start,
                 verbose=verbose)

    ρ .= reshape(exp.(result.solution), (ngrid..., nbeads))

    if verbose
        H = free_energy(system, ρ)
        err = isempty(result.history) ? NaN : result.history[end]
        msg = result.idid ? "DFT converged" : "DFT did not converge"
        @info "$msg after $(iter_count[]) iterations: err = $(round(err; sigdigits=3)) | F = $(round(H; sigdigits=6))"
    end
end

"""
    get_new_profile!(system::SCFTSystem, ρ, w, caches)

One SCFT fixed-point step: given the current field `w`, propagate the chains,
compute single-chain partition functions, assemble the new density profile
`ρ`, and evaluate the new field guess `caches.w_new` from that density. Bundles the
per-iteration sequence (exp-field precompute → propagate → partition
functions → densities → fields) that `scft_iterate!` previously inlined,
mirroring how the generic `get_new_profile!` above bundles `δFδρ_res! →
evaluate_external_field! → propagate!` into one fixed-point map — both take the same
`(system, ρ, field, caches)` shape.

`caches` is a named tuple of buffers preallocated once outside the iteration
loop: `w_new, w_bulk, dz, cache_external, q_fwd, q_bwd, buf_r, buf_c, P, iP,
weights, V_eff, exp_field, inv_exp_field, scratch`.

Returns `Q` (one partition function per molecule type).
"""
function get_new_profile!(system::SCFTSystem, ρ, w, caches)
    (; w_new, w_bulk, dz, cache_external, q_fwd, q_bwd, buf_r, buf_c, P, iP,
       weights, V_eff, exp_field, inv_exp_field, scratch) = caches
    nd = dimension(system)
    FT = eltype(ρ)

    # exp_field[α] = exp(w_bulk[α] - w_α); reused by propagate!,
    # compute_partition_functions, and compute_densities! below.
    for α in eachindex(exp_field)
        w_α = selectdim(w, nd+1, α)
        @. exp_field[α]     = exp(w_bulk[α] - w_α)
        @. inv_exp_field[α] = one(FT) / exp_field[α]
    end

    cache_propagator = (q_fwd, q_bwd, buf_r, buf_c, P, iP)
    propagate!(system, ρ, w, cache_propagator;
              w_bulk=w_bulk, exp_field=exp_field)
    Q = compute_partition_functions(system, w, w_bulk, q_fwd, dz;
                               weights=weights, V_eff=V_eff, exp_field=exp_field)
    compute_densities!(system, w, w_bulk, q_fwd, q_bwd, Q, ρ;
                       V_eff=V_eff, exp_field=exp_field, inv_exp_field=inv_exp_field)
    compute_fields!(system, ρ, w_new; scratch=scratch)
    # No-ops today (system.external_field is always nothing — see SCFTSystem's docstring);
    # mirrors the generic get_new_profile!'s δFδρ_res! → evaluate_external_field! →
    # propagate! ordering (bulk field first, external field added on top) so a future
    # field-type implementation for SCFTLatticeFluid only needs to add the dispatch, not
    # touch this loop.
    evaluate_external_field!(system, ρ, w_new, cache_external)

    return Q
end

"""
    converge!(system::SCFTSystem, ρ; kwargs...)

SCFT iteration loop: Picard mixing followed by Anderson acceleration, via `aasol`
(`src/utils/anderson.jl`) — the same generic Anderson-acceleration engine the generic
`converge!` above uses. `get_new_profile!` (above) is the fixed-point map
`w -> w_new(ρ(w))`; unlike the generic loop's `ln(ρ)`-space map (a positivity-preserving
trick specific to density), no log-space transform is needed here — the SCFT field `w` is
unconstrained in sign, so `aasol` iterates on it directly.

Stability condition (Picard): `beta < 2 / (1 + kappa)`.
For kappa=20 the default beta=0.01 gives ~10× safety margin.

`anderson_start` (switch from Picard to Anderson once the residual drops below this) is
translated to `aasol`'s `picard_atol` (with `picard_rtol=0` and a generous `picard_maxit`,
since the residual check triggers the switch in practice well before that iteration cap).

# Keyword Arguments
- `maxit::Int=5000`: Maximum Anderson-phase iterations (passed to `aasol` as both `maxit`
  and `picard_maxit`, since the Picard phase is expected to exit via `anderson_start`
  well before either cap).
- `beta::Float64=0.01`: Mixing coefficient for both Picard and Anderson steps.
- `tol::Float64=1e-6`: Convergence tolerance (`aasol`'s `atol`, 2-norm residual).
- `anderson_start::Float64=1e-2`: Switch from Picard to Anderson when the residual norm
  drops below this (`aasol`'s `picard_atol`).
- `anderson_m::Int=5`: Anderson history length (number of past iterates kept).
- `quadrature::Symbol=:trapz`: Quadrature rule for partition-function integrals.
  `:trapz` — periodic composite trapezoidal (uniform weights, spectral convergence for
  smooth periodic functions, works for any N). Recommended for periodic SCFT domains.
  `:simpson` — composite Simpson rule (O(h⁴), requires odd N per dimension).
- `log_interval::Int=100`: Log every N iterations (0 = never).
- `save_interval::Int=0`: Call `save_callback` every N iterations (0 = never).
- `save_callback`: `f(iter, ρ_array, w_array)` called at each save interval.
- `verbose::Bool=true`: Print convergence summary (forwarded to `aasol`'s own
  per-iteration `verbose` logging too).
"""
function converge!(system::SCFTSystem, ρ;
    maxit          :: Int     = 5000,
    beta                     = 0.01,
    tol                      = 1e-6,
    anderson_start           = 1e-2,
    anderson_m     :: Int     = 5,
    quadrature     :: Symbol  = :trapz,
    log_interval   :: Int     = 100,
    save_interval  :: Int     = 0,
    save_callback             = nothing,
    verbose        :: Bool    = false,
)
    dz = structure_dz(system.structure)
    FT = eltype(ρ)

    # Universal preallocation entry point (src/base/devices.jl), same as the generic
    # DFT-family converge! above: dispatches to preallocate_model,
    # preallocate_external_potential, and preallocate_propagator. SCFT's main field
    # buffer `w` is the returned δfδρ_res; `quadrature` is forwarded to preallocate_model
    # since it's a solve-time choice, not a system invariant.
    w, cache_model, cache_external, cache_propagator = preallocate(system, ρ; quadrature=quadrature)
    (; w_new, weights, V_eff, w_bulk, scratch, exp_field, inv_exp_field) = cache_model
    q_fwd, q_bwd, buf_r, buf_c, P, iP = cache_propagator

    compute_fields!(system, ρ, w; scratch=scratch)

    # Bundle the buffers preallocated above into a single cache passed to
    # get_new_profile! every iteration.
    caches = (; w_new, w_bulk, dz, cache_external, q_fwd, q_bwd, buf_r, buf_c, P, iP,
              weights, V_eff, exp_field, inv_exp_field, scratch)

    iter_count = Ref(0)
    Q_last = Ref{Vector{FT}}(FT[])

    # aasol's fixed-point map: given the current field (flattened), advance one SCFT
    # step and return the new field (flattened). ρ/w_new/Q_last are mutated as a side
    # effect, matching get_new_profile!'s own in-place contract.
    function GFix!(G, xin)
        copyto!(w, reshape(xin, size(w)))
        Q = get_new_profile!(system, ρ, w, caches)
        Q_last[] = Q
        iter_count[] += 1

        if log_interval > 0 && iter_count[] % log_interval == 0
            H = free_energy(system, ρ, w, Q; V_eff=V_eff, w_bulk=w_bulk)
            @info "SCFT iter $(lpad(iter_count[], 5)) | F = $(round(H; sigdigits=6))"
        end
        if save_interval > 0 && iter_count[] % save_interval == 0 && save_callback !== nothing
            save_callback(iter_count[], Array(ρ), Array(w))
        end

        copyto!(G, vec(w_new))
        return G
    end

    m = anderson_m
    N_flat = length(w)
    Vstore = similar(vec(w), N_flat, 3*m + 3)   # per aasol's docs: 3m+3 recommended width
    x0 = vec(copy(w))

    result = aasol(GFix!, x0, m, Vstore;
                   maxit = maxit, picard_maxit = maxit,
                   rtol = zero(FT), atol = FT(tol),
                   beta = FT(beta), picard_beta = FT(beta),
                   picard_rtol = zero(FT), picard_atol = FT(anderson_start),
                   verbose = verbose)

    w .= reshape(result.solution, size(w))
    # ρ already holds the density from the last GFix! call — aasol evaluates the
    # fixed-point map once more at `solution` to populate `functionval`, so ρ is
    # already consistent with the converged w.
    converged = result.idid
    err = isempty(result.history) ? FT(NaN) : FT(result.history[end])

    if verbose
        H = free_energy(system, ρ, w, Q_last[]; V_eff=V_eff, w_bulk=w_bulk)
        msg = converged ? "SCFT converged" : "SCFT did not converge"
        @info "$msg after $(iter_count[]) iterations: err = $(round(err; sigdigits=3)) | F = $(round(H; sigdigits=6))"
    end
end

export converge!