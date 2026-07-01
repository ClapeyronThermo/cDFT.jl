import Clapeyron: getsites

# ── GPU/Enzyme-compatible association support ───────────────────────────────

"""
    pack_assoc_params(model, HSd) -> NamedTuple

Pack Clapeyron's sparse association parameters into flat GPU-compatible vectors.
Returns flat vectors for each association pair and per-site data.
"""
function pack_assoc_params(model, HSd)
    sites = Clapeyron.getsites(model)
    n_sites_per_comp = [length(sites.n_sites[i]) for i in 1:length(model)]
    total_sites      = sum(n_sites_per_comp)
    n_sites_cumsum   = cumsum([0; n_sites_per_comp])  # length NC+1

    assoc_icomp = Int[];   assoc_jcomp = Int[]
    assoc_isite = Int[];   assoc_jsite = Int[]
    assoc_eps   = Float64[]; assoc_kap = Float64[]
    assoc_sig3  = Float64[]; assoc_dij = Float64[]

    eps_vals = model.params.epsilon_assoc.values
    kap_vals = model.params.bondvol.values
    sigma    = model.params.sigma.values

    for idx in 1:length(eps_vals.values)
        i, j = eps_vals.outer_indices[idx]
        a, b = eps_vals.inner_indices[idx]
        push!(assoc_icomp, i); push!(assoc_jcomp, j)
        push!(assoc_isite, a); push!(assoc_jsite, b)
        push!(assoc_eps, eps_vals.values[idx])
        push!(assoc_kap, kap_vals.values[idx])
        push!(assoc_sig3, sigma[i,j]^3)
        di = HSd[i]; dj = HSd[j]
        push!(assoc_dij, (di*dj)/(di+dj))
    end

    # n_sites_flat[global_site_idx] = number of sites of that type
    n_sites_flat = Int[]
    for i in 1:length(model)
        ns_i = sites.n_sites[i]
        for a in 1:length(ns_i)
            push!(n_sites_flat, ns_i[a])
        end
    end

    return (assoc_icomp, assoc_jcomp, assoc_isite, assoc_jsite,
            assoc_eps, assoc_kap, assoc_sig3, assoc_dij,
            n_sites_flat, n_sites_cumsum, total_sites)
end

"""
    pack_assoc_params_gc(model, HSd)

Like `pack_assoc_params` but for group-contribution models (HeterogcPCPSAFT, SAFTgammaMie)
where `epsilon_assoc.outer_indices` are *species* (molecular) indices rather than group/bead
indices.  Uses `getsites(model).site_translator[i][a][1]` to map each (species i, site a) to
the global bead/group index, then builds `n_sites_cumsum` and `n_sites_flat` over BEADS.

Returns bead indices in `assoc_icomp/jcomp` (for n₀ indexing in `f_assoc`) and species
indices in `assoc_ispec/jspec` (needed by SAFTgammaMie `_assoc_delta` for Tr via
`params.epsilon_species`).
"""
function pack_assoc_params_gc(model, HSd)
    sites    = Clapeyron.getsites(model)
    sigma    = model.params.sigma.values
    eps_vals = model.params.epsilon_assoc.values
    kap_vals = model.params.bondvol.values

    assoc_icomp = Int[]; assoc_jcomp = Int[]
    assoc_ispec = Int[]; assoc_jspec = Int[]
    assoc_isite = Int[]; assoc_jsite = Int[]
    assoc_eps   = Float64[]; assoc_kap = Float64[]
    assoc_sig3  = Float64[]; assoc_dij  = Float64[]

    nc_groups       = length(HSd)
    bead_site_count = zeros(Int, nc_groups)
    bead_local_idx  = Dict{Tuple{Int,Int}, Int}()
    site_to_bead    = Dict{Tuple{Int,Int}, Int}()

    for idx in 1:length(eps_vals.values)
        i, j = eps_vals.outer_indices[idx]
        a, b = eps_vals.inner_indices[idx]
        k = sites.site_translator[i][a][1]
        l = sites.site_translator[j][b][1]
        if !haskey(site_to_bead, (i, a))
            site_to_bead[(i, a)] = k
            bead_site_count[k] += 1
            bead_local_idx[(i, a)] = bead_site_count[k]
        end
        if !haskey(site_to_bead, (j, b))
            site_to_bead[(j, b)] = l
            bead_site_count[l] += 1
            bead_local_idx[(j, b)] = bead_site_count[l]
        end
    end

    n_sites_cumsum = cumsum([0; bead_site_count])
    n_sites_flat   = zeros(Int, max(1, sum(bead_site_count)))
    for ((i, a), k) in site_to_bead
        gidx = n_sites_cumsum[k] + bead_local_idx[(i, a)]
        n_sites_flat[gidx] = sites.n_sites[i][a]
    end
    total_sites = sum(bead_site_count)

    for idx in 1:length(eps_vals.values)
        i, j = eps_vals.outer_indices[idx]
        a, b = eps_vals.inner_indices[idx]
        k = sites.site_translator[i][a][1]
        l = sites.site_translator[j][b][1]
        push!(assoc_icomp, k);  push!(assoc_jcomp, l)
        push!(assoc_ispec, i);  push!(assoc_jspec, j)
        push!(assoc_isite, bead_local_idx[(i, a)])
        push!(assoc_jsite, bead_local_idx[(j, b)])
        push!(assoc_eps,   eps_vals.values[idx])
        push!(assoc_kap,   kap_vals.values[idx])
        push!(assoc_sig3,  sigma[k, l]^3)
        dk = HSd[k]; dl = HSd[l]
        push!(assoc_dij, (dk*dl)/(dk+dl))
    end

    return (assoc_icomp, assoc_jcomp, assoc_ispec, assoc_jspec,
            assoc_isite, assoc_jsite,
            assoc_eps, assoc_kap, assoc_sig3, assoc_dij,
            n_sites_flat, n_sites_cumsum, total_sites)
end

# Dispatch helper: zero-cost when model has no association (hasfield is compile-time constant)
@inline _assoc_or_zero(M, kk, n, params::P, T, vNC, vND) where P =
    hasfield(P, :assoc_eps) ? f_assoc(M, kk, n, params, T, vNC, vND,
                                      params.assoc_n_pairs, params.assoc_n_sites) : 0.0

# ── GPU/Enzyme-compatible kernel helpers ────────────────────────────────────

# GPU-safe NTuple indexing with a runtime integer.
# Julia's default NTuple getindex with a runtime Int calls jl_get_nth_field_checked,
# which is unsupported in PTX kernels.  This @generated helper expands to an explicit
# ifelse chain so the GPU compiler sees only conditional moves — no dynamic dispatch.
@generated function _nti(t::NTuple{N,T}, i::Int) where {N,T}
    ex = :(t[1])
    for k in 2:N
        ex = :(ifelse(i == $k, t[$k], $ex))
    end
    ex
end

# Default Wertheim association strength Δ: g_hs-based formula.
# Used by PCSAFT, PCPSAFT, QPCPSAFT, HomogcPCPSAFT (all same Δ formula).
# Model-specific overrides (e.g. SAFTVRMieModel) use a more specific M dispatch.
# The n, kk, NC, ND, M arguments are unused here but required for a uniform signature.
@inline function _assoc_delta(p, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix,
                               ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M}
    p > n_pairs && return 0.0
    dij_p  = params.assoc_dij[p]
    inv1n3 = 1.0 / (1.0 - n3_mix)
    g_hs   = inv1n3 + 0.5*dij_p*xi_mix*n2_mix*inv1n3^2 +
             dij_p*dij_p*n2_mix*n2_mix*xi_mix*(inv1n3^3)/18.0
    return g_hs * params.assoc_sig3[p] * expm1(params.assoc_eps[p]/T) * params.assoc_kap[p]
end

# ── @generated tuple-construction helpers ────────────────────────────────────
#
# f_assoc(::Val{NP}) is a regular @inline (not @generated) so Enzyme can inline
# and trace it without hitting the runtime_generic_fwd JIT path. But closures
# (ntuple ... do) box captured mutable variables on the GPU heap. We avoid
# closures entirely by delegating tuple construction to these @generated helpers,
# which generate flat, closure-free, GPU-safe code. Mutated scalars (n3_mix etc.)
# are passed as arguments — not captured — so no boxing occurs.

# Per-component number density n₀ᵢ = ρᵢ/dᵢ = n[kk,2,i]/HSd[i]
@generated function _assoc_n0(n, kk, params, ::Val{NC}) where NC
    stmts = [:($(Symbol("_n0_$i")) = n[kk, 2, $i] / params.HSd[$i]) for i in 1:NC]
    return quote
        $(stmts...)
        tuple($([ Symbol("_n0_$i") for i in 1:NC ]...))
    end
end

# Per-component geometry factor ξᵢ = 1 − |nv2ᵢ|²/n2ᵢ²  (FV=4 always)
@generated function _assoc_xi(n, kk, params, ::Val{NC}, ::Val{ND}) where {NC, ND}
    stmts = Expr[]
    for i in 1:NC
        n2v  = Symbol("_xi_n2_$i")
        sqv  = Symbol("_xi_sq_$i")
        xiv  = Symbol("_xi_$i")
        push!(stmts, :($n2v = π * n[kk, 2, $i] * params.HSd[$i]))
        push!(stmts, :($sqv = 0.0))
        if ND >= 1
            ndv = Symbol("_xi_nd_$(i)_1")
            push!(stmts, :($ndv = -2.0*π * n[kk, 4, $i]))
            push!(stmts, :($sqv += $ndv * $ndv))
        end
        if ND >= 2
            ndv = Symbol("_xi_nd_$(i)_2")
            push!(stmts, :($ndv = -2.0*π * n[kk, 5, $i]))
            push!(stmts, :($sqv += $ndv * $ndv))
        end
        if ND >= 3
            ndv = Symbol("_xi_nd_$(i)_3")
            push!(stmts, :($ndv = -2.0*π * n[kk, 6, $i]))
            push!(stmts, :($sqv += $ndv * $ndv))
        end
        push!(stmts, :($xiv = 1.0 - $sqv / ($n2v * $n2v)))
    end
    return quote
        $(stmts...)
        tuple($([ Symbol("_xi_$i") for i in 1:NC ]...))
    end
end

# NP explicit Δ evaluations; n3_mix/n2_mix/xi_mix passed as args to avoid boxing
@generated function _assoc_delta_vals(n, params, T, kk, n3_mix, n2_mix, xi_mix,
                                       ::Val{NC}, ::Val{ND}, ::Val{NP}, ::Type{M}) where {NC, ND, NP, M}
    stmts = [:($(Symbol("_Δ$p")) = _assoc_delta($p, $NP, n, params, T, kk,
                                                  n3_mix, n2_mix, xi_mix,
                                                  Val($NC), Val($ND), M))
             for p in 1:NP]
    return quote
        $(stmts...)
        tuple($([ Symbol("_Δ$p") for p in 1:NP ]...))
    end
end

# Initial X₀ = (0.9, …, 0.9) with NS elements
@generated function _assoc_X0(::Val{NS}) where NS
    :(tuple($(fill(0.9, NS)...)))
end

# ── @generated successive-substitution step ─────────────────────────────────
#
# Generates exactly NS explicit site updates with literal pair-index p — no
# closures, no runtime NTuple indexing for the outer pair loop. GPU-safe.
@generated function _assoc_SS_step(X::NTuple{NS,Float64},
                                    n0::NTuple{NC,Float64},
                                    xi::NTuple{NC,Float64},
                                    Δ_vals::NTuple{NP,Float64},
                                    params) where {NS, NC, NP}
    stmts = Expr[]
    for s in 1:NS
        denom_parts = Expr[]
        for p in 1:NP
            push!(denom_parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                params.assoc_n_jb[$p] *
                _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                Δ_vals[$p] * _nti(X, params.assoc_jb_global[$p]),
                0.0)))
            push!(denom_parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                params.assoc_n_ia[$p] *
                _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                Δ_vals[$p] * _nti(X, params.assoc_ia_global[$p]),
                0.0)))
        end
        denom_expr = reduce((a, b) -> :($a + $b), denom_parts)
        new_x = Symbol("new_x_$s")
        blend = Symbol("blend_$s")
        push!(stmts, :($new_x = 1.0 / (1.0 + $denom_expr)))
        push!(stmts, :($blend = 0.5 * (X[$s] + $new_x)))
    end
    result = :(tuple($([ Symbol("blend_$s") for s in 1:NS ]...)))
    return quote
        $(stmts...)
        $result
    end
end

# ── @generated safeguarded Newton-Raphson step ──────────────────────────────
#
# Builds the NS×NS Wertheim Jacobian in flat scalar code (no closures, GPU-safe),
# solves J*δX = F via Gaussian elimination (no pivoting; J is diagonally dominant
# near the fixed point since J_{ss} = 1 + C_s ≥ 1), then applies the update.
# If X_s - δX_s ∉ (0,1) for any site, that site falls back to the unblended SS
# update — same safeguard as Clapeyron's assoc_matrix_solve_general.
# Called FROM the regular @inline f_assoc so Enzyme traces f_assoc directly.
@generated function _assoc_newton_step(X::NTuple{NS,Float64},
                                        n0::NTuple{NC,Float64},
                                        xi::NTuple{NC,Float64},
                                        Δ_vals::NTuple{NP,Float64},
                                        params) where {NS, NC, NP}
    stmts = Expr[]
    cnt = Ref(0)
    new_sym() = (cnt[] += 1; Symbol("_nt_$(cnt[])"))

    # Step A: per-site denominator C_s and unblended SS fallback x_ss_s
    C_syms  = [new_sym() for _ in 1:NS]
    ss_syms = [new_sym() for _ in 1:NS]
    for s in 1:NS
        parts = Expr[]
        for p in 1:NP
            push!(parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                params.assoc_n_jb[$p] *
                _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                Δ_vals[$p] * _nti(X, params.assoc_jb_global[$p]), 0.0)))
            push!(parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                params.assoc_n_ia[$p] *
                _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                Δ_vals[$p] * _nti(X, params.assoc_ia_global[$p]), 0.0)))
        end
        C_expr = reduce((a, b) -> :($a + $b), parts)
        push!(stmts, :($(C_syms[s]) = $C_expr))
        push!(stmts, :($(ss_syms[s]) = 1.0 / (1.0 + $(C_syms[s]))))
    end

    # Step B: residuals F_s = X_s*(1 + C_s) - 1
    F_syms = [new_sym() for _ in 1:NS]
    for s in 1:NS
        push!(stmts, :($(F_syms[s]) = X[$s] * (1.0 + $(C_syms[s])) - 1.0))
    end

    # Step C: Jacobian J_{st} = (1+C_s)*δ_{st} + X_s * ∂C_s/∂X_t
    J_sym = [new_sym() for s in 1:NS, t in 1:NS]
    for s in 1:NS, t in 1:NS
        k_parts = Expr[]
        for p in 1:NP
            push!(k_parts, :(ifelse(params.assoc_ia_global[$p] == $s,
                ifelse(params.assoc_jb_global[$p] == $t,
                    params.assoc_n_jb[$p] *
                    _nti(n0, params.assoc_jcomp[$p]) * _nti(xi, params.assoc_jcomp[$p]) *
                    Δ_vals[$p], 0.0), 0.0)))
            push!(k_parts, :(ifelse(params.assoc_jb_global[$p] == $s,
                ifelse(params.assoc_ia_global[$p] == $t,
                    params.assoc_n_ia[$p] *
                    _nti(n0, params.assoc_icomp[$p]) * _nti(xi, params.assoc_icomp[$p]) *
                    Δ_vals[$p], 0.0), 0.0)))
        end
        K_st  = reduce((a, b) -> :($a + $b), k_parts)
        diag  = s == t ? :(1.0 + $(C_syms[s])) : :(0.0)
        push!(stmts, :($(J_sym[s,t]) = $diag + X[$s] * $K_st))
    end

    # Step D: Gaussian elimination on augmented matrix [J | F]
    J_aug = Matrix{Symbol}(undef, NS, NS + 1)
    for s in 1:NS
        for t in 1:NS; J_aug[s,t] = J_sym[s,t]; end
        J_aug[s, NS+1] = F_syms[s]
    end
    for k in 1:NS
        for i in k+1:NS
            f = new_sym()
            push!(stmts, :($f = $(J_aug[i,k]) / $(J_aug[k,k])))
            for j in k+1:NS+1
                t = new_sym()
                push!(stmts, :($t = $(J_aug[i,j]) - $f * $(J_aug[k,j])))
                J_aug[i,j] = t
            end
        end
    end

    # Step E: back substitution → δX
    δX = Vector{Symbol}(undef, NS)
    for k in NS:-1:1
        rhs = J_aug[k, NS+1]
        for j in k+1:NS
            t = new_sym()
            push!(stmts, :($t = $rhs - $(J_aug[k,j]) * $(δX[j])))
            rhs = t
        end
        δX[k] = new_sym()
        push!(stmts, :($(δX[k]) = $rhs / $(J_aug[k,k])))
    end

    # Step F: accept Newton if X_s - δX_s ∈ (0,1), else fall back to SS
    res = [new_sym() for _ in 1:NS]
    for s in 1:NS
        cand = new_sym()
        push!(stmts, :($cand = X[$s] - $(δX[s])))
        push!(stmts, :($(res[s]) = ifelse($cand > 0.0,
            ifelse($cand < 1.0, $cand, $(ss_syms[s])),
            $(ss_syms[s]))))
    end

    return quote
        $(stmts...)
        tuple($(res...))
    end
end

"""
Wertheim association free energy density at grid point `kk`.

The `::Val{1}` specialisation uses the analytical quadratic formula — pure scalar
arithmetic, fully GPU-safe with Enzyme forward mode. Dispatch via `params.assoc_n_pairs`
which is stored as `Val(nn)` in preallocate_params.

The `::Val{NP}` general method runs `5*NS` relaxed SS warm-up then `5*NS` Newton steps.
`f_assoc` is a regular `@inline` (not `@generated`) so Enzyme can inline and trace it
without hitting `runtime_generic_fwd`. NTuple construction uses `@generated` helpers
(`_assoc_n0`, `_assoc_xi`, `_assoc_delta_vals`, `_assoc_X0`, `_assoc_newton_step`) so
no closures appear in `f_assoc` itself — closures that capture mutated locals cause GPU
heap allocation.

`_assoc_delta` is dispatched on `::Type{M}` so each model can supply its own Δ formula.
"""
@inline function f_assoc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND},
                          ::Val{1}, ::Val{NS}) where {NC, ND, NS, M}
    F2    = 2
    FV    = F2 + 2

    # Mixture FMT densities
    n2_mix    = 0.0
    n3_mix    = 0.0
    nv2sq_mix = 0.0
    @inbounds for i in 1:NC
        nim = n[kk, F2, i] * params.m[i]
        n2_mix += π * nim * params.HSd[i]
        n3_mix += n[kk, F2+1, i] * params.m[i]
    end
    if ND >= 1
        nv2d = 0.0
        @inbounds for i in 1:NC; nv2d -= 2.0*π * n[kk, FV,   i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 2
        nv2d = 0.0
        @inbounds for i in 1:NC; nv2d -= 2.0*π * n[kk, FV+1, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 3
        nv2d = 0.0
        @inbounds for i in 1:NC; nv2d -= 2.0*π * n[kk, FV+2, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    xi_mix = 1.0 - nv2sq_mix / (n2_mix*n2_mix)

    # ── Analytical 1-pair solution ─────────────────────────────────────────
    ic  = params.assoc_icomp[1];  jc  = params.assoc_jcomp[1]

    Δ_val = _assoc_delta(1, 1, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)

    n0_ic    = n[kk, F2, ic] / params.HSd[ic]
    n2_ic    = π * n[kk, F2, ic] * params.HSd[ic]
    nv2sq_ic = 0.0
    if ND >= 1; nd = -2.0*π*n[kk, FV,   ic]; nv2sq_ic += nd*nd; end
    if ND >= 2; nd = -2.0*π*n[kk, FV+1, ic]; nv2sq_ic += nd*nd; end
    if ND >= 3; nd = -2.0*π*n[kk, FV+2, ic]; nv2sq_ic += nd*nd; end
    xi_ic = 1.0 - nv2sq_ic / (n2_ic*n2_ic)

    n0_jc    = n[kk, F2, jc] / params.HSd[jc]
    n2_jc    = π * n[kk, F2, jc] * params.HSd[jc]
    nv2sq_jc = 0.0
    if ND >= 1; nd = -2.0*π*n[kk, FV,   jc]; nv2sq_jc += nd*nd; end
    if ND >= 2; nd = -2.0*π*n[kk, FV+1, jc]; nv2sq_jc += nd*nd; end
    if ND >= 3; nd = -2.0*π*n[kk, FV+2, jc]; nv2sq_jc += nd*nd; end
    xi_jc = 1.0 - nv2sq_jc / (n2_jc*n2_jc)

    # Precomputed site counts (NP=1 → literal index 1)
    n_ia = params.assoc_n_ia[1]
    n_jb = params.assoc_n_jb[1]

    kia = n_ia * n0_ic * xi_ic * Δ_val
    kjb = n_jb * n0_jc * xi_jc * Δ_val
    _b  = 1.0 - kia + kjb
    X_ia = 2.0 / (_b + sqrt(_b*_b + 4.0*kia))
    X_jb = 1.0 / (1.0 + kia * X_ia)

    return n0_ic * xi_ic * n_ia * (Base.log(abs(X_ia)) - 0.5*X_ia + 0.5) +
           n0_jc * xi_jc * n_jb * (Base.log(abs(X_jb)) - 0.5*X_jb + 0.5)
end

@inline function f_assoc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND},
                          ::Val{NP}, ::Val{NS}) where {NC, ND, NP, NS, M}
    it_ss     = 5 * NS   # compile-time constants (NS is a type parameter)
    it_newton = 5 * NS
    F2 = 2; FV = 4

    # Mixture FMT densities (straight-line loops, no closures)
    n2_mix = 0.0; n3_mix = 0.0; nv2sq_mix = 0.0
    @inbounds for i in 1:NC
        nim = n[kk, F2, i] * params.m[i]
        n2_mix += π * nim * params.HSd[i]
        n3_mix += n[kk, F2+1, i] * params.m[i]
    end
    if ND >= 1
        nv2d = 0.0
        @inbounds for i in 1:NC; nv2d -= 2.0*π * n[kk, FV,   i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 2
        nv2d = 0.0
        @inbounds for i in 1:NC; nv2d -= 2.0*π * n[kk, FV+1, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    if ND >= 3
        nv2d = 0.0
        @inbounds for i in 1:NC; nv2d -= 2.0*π * n[kk, FV+2, i] * params.m[i]; end
        nv2sq_mix += nv2d * nv2d
    end
    xi_mix = 1.0 - nv2sq_mix / (n2_mix * n2_mix)

    # Per-component n₀ and ξ via @generated helpers (no closures, no boxing)
    n0     = _assoc_n0(n, kk, params, Val(NC))
    xi     = _assoc_xi(n, kk, params, Val(NC), Val(ND))
    # n3_mix/n2_mix/xi_mix passed as args (not captured) → no boxing
    Δ_vals = _assoc_delta_vals(n, params, T, kk, n3_mix, n2_mix, xi_mix,
                                Val(NC), Val(ND), Val(NP), M)

    # 5*NS relaxed SS warm-up (gets X into convergence basin) + 5*NS Newton
    X = _assoc_X0(Val(NS))
    for _ in 1:it_ss
        X = _assoc_SS_step(X, n0, xi, Δ_vals, params)
    end
    for _ in 1:it_newton
        X = _assoc_newton_step(X, n0, xi, Δ_vals, params)
    end

    # Accumulate f_assoc = Σᵢ Σₐ n₀ᵢ ξᵢ nᵢₐ (ln Xᵢₐ - Xᵢₐ/2 + 1/2)
    # Regular for-loops + _nti: no closures, GPU-safe
    res = 0.0
    for i in 1:NC
        base_i = _nti(params.n_sites_cumsum, i)
        ns_i   = _nti(params.n_sites_cumsum, i + 1) - base_i
        n0i    = _nti(n0, i)
        xii    = _nti(xi, i)
        for a in 1:ns_i
            s        = base_i + a
            n_ia_val = _nti(params.n_sites_flat, s)
            X_val    = _nti(X, s)
            res     += n0i * xii * n_ia_val * (Base.log(abs(X_val)) - 0.5*X_val + 0.5)
        end
    end
    res
end
