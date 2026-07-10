include("eos.jl")
"""
    SCFTSystem(model, structure, options=DFTOptions(); mol_structure, ensemble, n_molecules)

Main SCFT system struct containing all information for SCFT calculations.

# Fields
- `model::EoSModel`: Bulk interaction model (e.g., `SCFTLatticeFluid`) — `model.components`
  are molecule types (chains and solvents, unified), `model.groups.flattenedgroups` are
  species.
- `species::DFTSpecies`: Per-molecule-type SCFT-calculation parameters (chain order,
  ensemble, counts), from `get_species(model, ...)`.
- `structure::DFTStructure`: Spatial discretization.
- `propagator::DFTPropagator`: Chain propagator (DiscreteGaussianChainPropagator).
- `options::DFTOptions`: Device and solver options.
- `external_field`: `nothing` or a `Vector{<:ExternalFieldModel}` — forward-compatible
  plumbing only; no `evaluate_external_field!(structure, field, model::SCFTLatticeFluid,
  ...)` method exists yet for any field type, so this must be `nothing` today.
"""
struct SCFTSystem{M<:EoSModel, S<:DFTSpecies, T<:DFTStructure, P<:DFTPropagator, O<:DFTOptions, EF} <: AbstractcDFTSystem
    model::M
    species::S
    structure::T
    propagator::P
    options::O
    external_field::EF
end
export SCFTSystem

# TwoPhase* structures carry a *second* bulk-density vector (`ρbulk2`) meant to seed a
# sigmoidal interface between two distinct bulk phases — validated against here, at
# construction time, rather than waiting for `initialize_profiles` to be called, since
# SCFT has no two-phase/interfacial-tension support at all (see
# `surface_tension(::SCFTSystem, ρ)`) and the mismatch is knowable immediately.

nspecies(x::SCFTSystem) = length(x.model.groups.flattenedgroups)

"""
    SCFTSystem(model::SCFTLatticeFluid, structure::DFTStructure, options::DFTOptions=DFTOptions();
               mol_structure, ensemble=fill(:grand_canonical,ncomponents), n_molecules=zeros(ncomponents))

Construct an `SCFTSystem`, following the same pattern as `DFTSystem(model, structure,
options)`: build the bulk `model`, build a `structure` (`Uniform1DCart`,
`LamellarStack1DCart`, `BCC3DCart`, ... — any `DFTStructure` except `TwoPhaseSystem`, which
this constructor rejects; SCFT has no two-phase/interfacial-tension support), then
combine them here.

`structure.ρbulk` (one entry per molecule type, i.e. `model.components`) supplies each
molecule type's **target total bulk density** (the total density contributed by every
segment of every instance of that molecule type — e.g. `rho0` for a pure single-component
melt) — used directly for `:grand_canonical` molecule types (matching DFT-family's
convention exactly: `ρbulk` there is likewise the per-component bulk density used both for
the physical bulk condition and for seeding `initialize_profiles`), and as an
`initialize_profiles`/morphology-seeding seed value only (not real physics) for
`:canonical` molecule types, which instead get their actual bulk condition from
`n_molecules`.

# Arguments
- `model`: An `EoSModel` for the bulk interaction (e.g., `SCFTLatticeFluid`), built from a
  Clapeyron group-contribution `grouplist` — `model.components` are molecule types
  (chains/solvents), `model.groups.flattenedgroups` are species.
- `structure`: Spatial discretization, e.g. `Uniform1DCart((0.,0.), ρbulk, [0.,L], ngrid)`.
- `options`: `DFTOptions` for device/solver settings.

# Keyword Arguments
- `mol_structure::Dict{String,<:MolStructure}`: Per molecule type, chain connectivity —
  the same mechanism `HeterogcPCPSAFT`/`SAFTgammaMie` use, e.g.
  `Dict("diblock" => custom_structure("AAAABBBB"), "solvent" => custom_structure("S"))`.
- `ensemble`: Per molecule type, `:canonical` or `:grand_canonical`. Defaults to
  `:grand_canonical` for every molecule type (`structure.ρbulk` alone then fully
  determines the bulk condition, matching DFT-family's convention).
- `n_molecules`: Per molecule type, chain/molecule count — only meaningful (and needed)
  for molecule types explicitly marked `:canonical`; defaults to zero for every type.
- `external_field`: `nothing` (default), a single `ExternalFieldModel`, or a
  `Vector{<:ExternalFieldModel}` — forward-compatible plumbing only (see the struct
  docstring above); passing anything other than `nothing` today has no effect since no
  field-type implementation exists yet for `SCFTLatticeFluid`.
"""
function SCFTSystem(model::EoSModel, structure::DFTStructure, options::DFTOptions=DFTOptions();
    mol_structure::Dict{String,<:MolStructure},
    ensemble::Vector{Symbol} = fill(:grand_canonical, length(model.components)),
    n_molecules::AbstractVector = zeros(length(model.components)),
    external_field = nothing,
)
    @assert length(structure.ρbulk) == length(model.components) "structure.ρbulk must have one entry per molecule type ($(length(model.components))), got $(length(structure.ρbulk))"
    structure.system_type isa TwoPhaseSystem && error(
        """
        SCFTSystem does not support TwoPhaseSystem structures — SCFT has no two-phase/
        interfacial-tension support yet (see surface_tension(::SCFTSystem, ρ)). Use a
        Uniform*Cart or morphology structure instead.
        """
    )

    # get_species reads off the expanded model (chain order baked into
    # model.groups.i_groups) but SCFTSystem itself keeps the original, unexpanded
    # `model` — SCFT's mean-field fields are per species-letter, not per chain-position
    # instance (see expand_model's docstring).
    expanded_model = expand_model(model, mol_structure)
    species = get_species(expanded_model, structure; ensemble=ensemble, n_molecules=n_molecules)

    FP = fptype(options)
    propagator = get_propagator(model, species, structure, options.device, FP)

    # Normalize a bare field to a Vector, matching DFTSystem's constructor convention
    # (src/base/base.jl:71,79).
    normalized_external_field = external_field isa ExternalFieldModel ? [external_field] : external_field

    return SCFTSystem(model, species, structure, propagator, options, normalized_external_field)
end

# Defined here (not src/base/show.jl) since SCFTSystem doesn't exist yet when that file is
# included from base/base.jl — matches the DFTSystem/DGTSystem show.jl pattern otherwise.
function Base.show(io::IO, ::MIME"text/plain", system::SCFTSystem)
    n = nspecies(system)
    print(io, "SCFTSystem with ", n, " species")
    n > 1 && print(io, "s")
    println(io, ":")
    print(io, " model: ")
    show(io, system.model)
    println(io)
    println(io, " molecule types: ", length(system.model.components))
    println(io, " structure: "*string(typeof(system.structure)))
    print(io, " device: "*string(typeof(system.options.device)))
end

"""
    compute_fields!(system::SCFTSystem, ρ, w)

Compute the mean-field potential fields `w` from the density profiles `ρ`, for local
Flory-Huggins interactions with Helfand compressibility:

    w_α(r) = Σ_β (χ_αβ / ρ₀) ρ_β(r) + (ζ / ρ₀)(ρ₊(r) / ρ₀ - 1)

where ρ₊ = Σ_α ρ_α is the total density, and `χ`/`ρ₀`/`ζ` come from `system.model`
(an `SCFTLatticeFluid`, `src/models/SCFT/eos.jl`).
"""
function compute_fields!(system::SCFTSystem, ρ, w; scratch=nothing)
    nd = dimension(system)
    nspecies_ = nspecies(system)
    FT = eltype(w)
    chi = FT.(system.model.params.chi.values)
    rho0 = FT(system.model.rho0)
    kappa = FT(system.model.kappa)

    # Use preallocated scratch if provided, otherwise allocate.
    # scratch must have the same shape/device as a single species slice of w.
    ρ_total = scratch !== nothing ? scratch : similar(selectdim(w, nd+1, 1))

    # Accumulate total density into scratch (in-place, no allocation)
    ρ_total .= zero(FT)
    for α in 1:nspecies_
        ρ_total .+= selectdim(ρ, nd+1, α)
    end

    # Overwrite ρ_total in-place with the compressibility term to avoid a second allocation:
    # comp_term = (ζ / ρ₀)(ρ₊ / ρ₀ - 1)
    @. ρ_total = (kappa / rho0) * (ρ_total / rho0 - one(FT))

    # Field for each species
    for α in 1:nspecies_
        w_α = selectdim(w, nd+1, α)
        w_α .= ρ_total
        for β in 1:nspecies_
            if chi[α, β] != zero(FT)
                w_α .+= (chi[α, β] / rho0) .* selectdim(ρ, nd+1, β)
            end
        end
    end
end

"""
    compute_bulk_fields(model::EoSModel, bulk_densities::Vector{Float64})

Compute bulk (uniform) fields from bulk densities. Returns a vector of field
values, one per species.
"""
function compute_bulk_fields(model::EoSModel, bulk_densities::AbstractVector{T}) where {T<:AbstractFloat}
    nspecies = length(bulk_densities)
    FT = eltype(bulk_densities)
    chi = FT.(model.params.chi.values)
    rho0 = FT(model.rho0)
    kappa = FT(model.kappa)

    ρ_total = sum(bulk_densities)
    comp_term = (kappa / rho0) * (ρ_total / rho0 - one(FT))

    w_bulk = zeros(FT, nspecies)
    for α in 1:nspecies
        w_bulk[α] = comp_term
        for β in 1:nspecies
            w_bulk[α] += (chi[α, β] / rho0) * bulk_densities[β]
        end
    end
    return w_bulk
end

"""
    compute_bulk_densities(system::SCFTSystem)

Return the bulk density of each species (`system.species.bulk_density`) — computed once,
ahead of time, in `get_species` when `system` was constructed (from
`structure.ρbulk`/`ensemble`/`n_molecules`, split per species by segment count and the
box's exact geometric volume). Kept as a function (rather than requiring callers to reach
into `system.species.bulk_density` directly) purely for API stability — this used to be
where the ensemble-aware split was actually computed, on every call.

Returns a vector of length `nspecies` with element type `Float64` (see `SCFTSpecies`).
"""
compute_bulk_densities(system::SCFTSystem) = system.species.bulk_density

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

Compute single-molecule partition functions from shifted propagators (chains and
solvents, unified — a solvent is just an `N=1` molecule type flowing through the same
propagator arrays):
    Q̃_c = (1/V_eff) ∫ q̃_fwd[c](:, N_c) dr

where `q̃` is the propagator computed with shifted fields `Δw = w - w_bulk`.
For a uniform system at bulk densities, `Q̃ ≈ 1`.

Returns `Q::Vector{Float64}`, one entry per molecule type (`system.model.components`).
"""
function compute_partition_functions(system::SCFTSystem, w, w_bulk, q_fwd, dz;
                                     weights=nothing, V_eff=nothing, exp_field=nothing)
    nd = dimension(system)
    sequence = system.species.sequence
    nmol = length(sequence)
    FT = fptype(system.options)

    V_eff = V_eff !== nothing ? FT(V_eff) : FT(effective_volume(system, dz))

    Q = Vector{FT}(undef, nmol)
    for c in 1:nmol
        Nc = length(sequence[c])
        if weights !== nothing
            # GPU-friendly: dot product with precomputed weight array — no host transfer
            Q[c] = sum(selectdim(q_fwd[c], nd+1, Nc) .* weights) / V_eff
        else
            # Periodic trapezoidal rule (matches `effective_volume`'s periodic convention,
            # since q_fwd lives on the periodic FFT grid — Simpson's `∫` assumes a
            # non-periodic domain with duplicated endpoints and is NOT consistent with
            # `dz = L/ngrid`/`V_eff = prod(dz)*prod(ngrid)` here).
            q_end = selectdim(q_fwd[c], nd+1, Nc)
            Q[c] = sum(q_end) * prod(dz) / V_eff
        end
    end

    return Q
end

"""
    compute_densities!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, Q, ρ)

Compute density profiles from shifted propagators and partition functions, for every
molecule type (chains and solvents, unified — a solvent is just an `N=1` molecule type,
for which this formula reduces exactly to the old separate solvent formulas):

    ρ_α(r) += prefactor * Σ_{s: α(s)=α} q̃_fwd(r,s) * q̃_bwd(r, N+1-s) * exp(Δw_α(r))

where `Δw = w - w_bulk`, and the exp(Δw) corrects for double-counting of the
Boltzmann weight at segment s. The shift factors cancel between numerator and
denominator (Q̃), keeping values near O(1). `prefactor = n_molecules/(V_eff*Q̃)`
(canonical) or `bulk_density/(N*Q̃)` (grand canonical).
"""
function compute_densities!(system::SCFTSystem, w, w_bulk, q_fwd, q_bwd, Q, ρ;
                            V_eff=nothing, exp_field=nothing, inv_exp_field=nothing)
    nd = dimension(system)
    species = system.species
    nmol = length(species.sequence)
    dz = structure_dz(system.structure)
    FT = eltype(ρ)

    V_eff = V_eff !== nothing ? FT(V_eff) : FT(effective_volume(system, dz))

    # Zero out densities
    ρ .= zero(FT)

    for c in 1:nmol
        seg_spec = species.sequence[c]
        Nc = length(seg_spec)
        Qc = Q[c]

        # Prefactor depends on ensemble
        if species.ensemble[c] == :canonical
            prefactor = FT(species.n_molecules[c]) / (V_eff * Qc)
        else
            prefactor = FT(species.molecule_bulk_density[c]) / (FT(Nc) * Qc)
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
end

"""
    free_energy(system::SCFTSystem, ρ, w, Q)

Compute the SCFT free energy (mean-field Hamiltonian):

    H = U_int + U_comp - Σ_K ∫w_K ρ_K dr - Σ_c n_c ln(Q̃_c) - Σ_c (bulk_density_c/N_c) V Q̃_c

summed over every molecule type (chains and solvents, unified — a solvent is just an
`N=1` molecule type). `Q` is the *shifted* partition function (Q̃) from propagators
computed with `Δw = w - w_bulk`. For grand-canonical molecule types, the fugacity/bulk
correction exactly cancels the propagator's `exp(±w_bulk_sum)` shift factor (verified by
direct re-derivation, not carried over from the old code — see below), so `Q̃` is used
directly with no extra correction; for canonical molecule types, the shift factor does
*not* cancel inside `log`, so it must be subtracted explicitly.
"""
function free_energy(system::SCFTSystem, ρ, w, Q;
                     V_eff=nothing, w_bulk=nothing)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    nspecies_ = nspecies(system)
    dz = structure_dz(system.structure)

    # Use V_eff and w_bulk from the iteration loop when provided, so the free energy
    # is computed consistently with the quadrature rule chosen for the iteration.
    # Fallback to recomputing from scratch (CPU Simpson) for standalone calls.
    if V_eff === nothing
        V_eff = effective_volume(system, dz)
    end
    if w_bulk === nothing
        w_bulk = compute_bulk_fields(system.model, compute_bulk_densities(system))
    end

    chi = system.model.params.chi.values
    rho0 = system.model.rho0
    kappa = system.model.kappa

    # Periodic composite-trapezoidal integral: sum(f)*prod(dz). Deliberately NOT the
    # general-purpose `∫` (Simpson's rule assuming non-periodic spacing dz=L/(N-1)) —
    # `structure_dz` returns the *periodic* spacing dz=L/N, so `∫` on a periodic grid
    # systematically undercounts by a factor (N-1)/N (the same discrepancy
    # `effective_volume`'s docstring already documents for volume/Q/density
    # normalization). Using the same periodic convention here keeps U_int/U_comp/wρ_sum
    # consistent with V_eff (and with each other) regardless of caller.
    per_∫(f) = sum(f) * prod(dz)

    # U_int = (1/ρ₀) ∫ Σ_{α<β} χ_αβ ρ_α ρ_β dr
    U_int_integrand = zeros(ngrid...)
    for α in 1:nspecies_
        for β in (α+1):nspecies_
            if chi[α, β] != 0.0
                U_int_integrand .+= chi[α, β] .* Array(selectdim(ρ, nd+1, α)) .* Array(selectdim(ρ, nd+1, β))
            end
        end
    end
    U_int = per_∫(U_int_integrand) / rho0

    # U_comp = (ζ / 2) ∫ (ρ₊/ρ₀ - 1)² dr
    # Consistent with w_comp = ζ/ρ₀ · (ρ₊/ρ₀ − 1) = δU_comp/δρ_α.
    ρ_total = zeros(ngrid...)
    for α in 1:nspecies_
        ρ_total .+= Array(selectdim(ρ, nd+1, α))
    end
    U_comp_integrand = (kappa / 2.0) .* (ρ_total ./ rho0 .- 1.0) .^ 2
    U_comp = per_∫(U_comp_integrand)

    # -Σ_K ∫ w_K ρ_K dr
    wρ_sum = 0.0
    for α in 1:nspecies_
        wρ_integrand = Array(selectdim(w, nd+1, α)) .* Array(selectdim(ρ, nd+1, α))
        wρ_sum += per_∫(wρ_integrand)
    end

    # Molecule-type contributions (chains and solvents, unified; a solvent is N_c=1).
    # Canonical: -n_c * ln(Q_c) where Q_c is the TRUE partition function.
    #   Q̃_c = Q_c * exp(Σ_s w_bulk[α(s)]), so ln(Q_c) = ln(Q̃_c) - Σ_s w_bulk[α(s)]
    #   -- the shift does NOT cancel inside log, so it must be subtracted explicitly.
    # Grand canonical: Ω = -kT V z Q_c, with z = bulk_density/(N_c * Q_c(bulk)) and
    #   Q_c(bulk) = exp(-Σ_s w_bulk[α(s)]) (uniform-field propagator). Substituting:
    #   Ω = -(bulk_density/N_c) V exp(Σ w_bulk) Q_c = -(bulk_density/N_c) V exp(Σ w_bulk) Q̃_c exp(-Σ w_bulk)
    #   the exp(±Σ w_bulk) factors cancel EXACTLY, leaving -(bulk_density/N_c) V Q̃_c with
    #   no correction at all -- unlike the canonical branch, this is *not* logarithmic.
    species = system.species
    molecule_sum = 0.0
    for c in eachindex(system.model.components)
        seg_spec = species.sequence[c]
        Nc = length(seg_spec)
        if species.ensemble[c] == :canonical
            w_bulk_sum = sum(w_bulk[seg_spec[s]] for s in 1:Nc)
            molecule_sum -= species.n_molecules[c] * (log(Q[c]) - w_bulk_sum)
        else
            molecule_sum -= (species.molecule_bulk_density[c] / Nc) * V_eff * Q[c]
        end
    end

    H = U_int + U_comp - wρ_sum + molecule_sum
    return H
end

"""
Guard-rails: `SCFTSystem <: AbstractcDFTSystem` makes it dispatch-reachable for these
generic, DFT-family-specific methods (`F_ideal`/`F_res`/`Clapeyron.VT_chemical_potential`
on an `EoSModel`, `system.species.nbeads`, `system.structure.ρbulk`, none of which
`SCFTLatticeFluid`/`SCFTSpecies` provide) — without these overrides they'd fail deep
inside with a confusing error instead of a clear one. Use
`free_energy(system::SCFTSystem, ρ, w, Q)` (above) for the SCFT free energy instead.
"""
free_energy(::SCFTSystem, ρ) = error(
    "free_energy(system::SCFTSystem, ρ) is not defined — SCFT free energy also depends " *
    "on the field w and partition functions Q. Use free_energy(system, ρ, w, Q) instead."
)
surface_tension(::SCFTSystem, ρ) = error(
    "surface_tension is not defined for SCFTSystem — SCFT systems are not vapor-liquid " *
    "interface calculations against an EoSModel bulk phase."
)
