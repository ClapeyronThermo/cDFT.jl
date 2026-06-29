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
    hasfield(P, :assoc_eps) ? f_assoc(M, kk, n, params, T, vNC, vND, params.assoc_n_pairs) : 0.0

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

# Fixed-point step for one site index `s` in the 20-site relaxed SS iteration.
# Defined as @inline (not a closure) so Enzyme can differentiate each element
# individually, avoiding the mixed-activity ntuple-closure issue.
@inline function _assoc_Xstep(s, X::NTuple{20,Float64}, total, n0, xi, Δ_vals,
                               params, n_pairs, ::Val{NC}) where NC
    if s > total
        return _nti(X, s)
    end
    comp_s = 1
    for cc in 1:NC
        if s <= _nti(params.n_sites_cumsum, cc + 1)
            comp_s = cc
            break
        end
    end
    local_s = s - _nti(params.n_sites_cumsum, comp_s)
    denom   = 1.0
    for p in 1:n_pairs
        ic_p = _nti(params.assoc_icomp, p); ia_p = _nti(params.assoc_isite, p)
        jc_p = _nti(params.assoc_jcomp, p); jb_p = _nti(params.assoc_jsite, p)
        if ic_p == comp_s && ia_p == local_s
            jb_g = _nti(params.n_sites_cumsum, jc_p) + jb_p
            denom += _nti(params.n_sites_flat, jb_g) * _nti(n0, jc_p) * _nti(xi, jc_p) *
                     _nti(X, jb_g) * _nti(Δ_vals, p)
        end
        if jc_p == comp_s && jb_p == local_s
            ia_g = _nti(params.n_sites_cumsum, ic_p) + ia_p
            denom += _nti(params.n_sites_flat, ia_g) * _nti(n0, ic_p) * _nti(xi, ic_p) *
                     _nti(X, ia_g) * _nti(Δ_vals, p)
        end
    end
    return 1.0 / denom
end

"""
Wertheim association free energy density at grid point `kk`.

The `::Val{1}` specialisation uses the analytical quadratic formula — pure scalar
arithmetic, fully GPU-safe with Enzyme reverse mode. Dispatch via `params.assoc_n_pairs`
which is stored as `Val(nn)` in preallocate_params.

The `::Val{NP}` general method (NP > 1 pairs) runs 50-iteration relaxed SS.

`_assoc_delta` is dispatched on `::Type{M}` so each model can supply its own Δ formula.
"""
@inline function f_assoc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}, ::Val{1}) where {NC, ND, M}
    F2    = 2
    FV    = F2 + 2

    # Mixture FMT densities (needed by default g_hs _assoc_delta; also by VRMie xi)
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
    ia  = params.assoc_isite[1];  jb  = params.assoc_jsite[1]

    Δ_val = _assoc_delta(1, 1, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)

    # Per-component n₀ᵢ = ρᵢ_molecular / dᵢ (NO m factor — Wertheim uses molecular density)
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

    n_ia = _nti(params.n_sites_flat, _nti(params.n_sites_cumsum, ic) + ia)
    n_jb = _nti(params.n_sites_flat, _nti(params.n_sites_cumsum, jc) + jb)

    # Self-consistency system (1 pair): kia*X_A² + (1-kia+kjb)*X_A - 1 = 0
    kia = n_ia * n0_ic * xi_ic * Δ_val
    kjb = n_jb * n0_jc * xi_jc * Δ_val
    _b  = 1.0 - kia + kjb
    X_ia = 2.0 / (_b + sqrt(_b*_b + 4.0*kia))
    X_jb = 1.0 / (1.0 + kia * X_ia)

    return n0_ic * xi_ic * n_ia * (Base.log(abs(X_ia)) - 0.5*X_ia + 0.5) +
           n0_jc * xi_jc * n_jb * (Base.log(abs(X_jb)) - 0.5*X_jb + 0.5)
end

@inline function f_assoc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}, ::Val{NP}) where {NC, ND, NP, M}
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

    # ── General case: 50-iteration successive substitution ─────────────────
    n_pairs = NP
    total   = params.total_sites

    # Per-component n₀ and ξ
    n0 = ntuple(i -> n[kk, F2, i] / params.HSd[i], Val(NC))
    xi = ntuple(Val(NC)) do i
        n2_i = π * n[kk, F2, i] * params.HSd[i]
        sq   = 0.0
        if ND >= 1; nd = -2.0*π*n[kk, FV,   i]; sq += nd*nd; end
        if ND >= 2; nd = -2.0*π*n[kk, FV+1, i]; sq += nd*nd; end
        if ND >= 3; nd = -2.0*π*n[kk, FV+2, i]; sq += nd*nd; end
        1.0 - sq / (n2_i*n2_i)
    end

    # Δ for each pair via model-dispatched _assoc_delta (explicit scalar calls for Enzyme)
    Δ1  = _assoc_delta(1,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ2  = _assoc_delta(2,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ3  = _assoc_delta(3,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ4  = _assoc_delta(4,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ5  = _assoc_delta(5,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ6  = _assoc_delta(6,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ7  = _assoc_delta(7,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ8  = _assoc_delta(8,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ9  = _assoc_delta(9,  n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ10 = _assoc_delta(10, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ11 = _assoc_delta(11, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ12 = _assoc_delta(12, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ13 = _assoc_delta(13, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ14 = _assoc_delta(14, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ15 = _assoc_delta(15, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ16 = _assoc_delta(16, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ17 = _assoc_delta(17, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ18 = _assoc_delta(18, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ19 = _assoc_delta(19, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ20 = _assoc_delta(20, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, Val(NC), Val(ND), M)
    Δ_vals = (Δ1, Δ2, Δ3, Δ4, Δ5, Δ6, Δ7, Δ8, Δ9, Δ10,
              Δ11, Δ12, Δ13, Δ14, Δ15, Δ16, Δ17, Δ18, Δ19, Δ20)

    # Initial X: literal tuple (Enzyme treats as constant)
    X = (0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)

    # Relaxed fixed-point α=0.5: X ← (X + F(X))/2.
    for _ in 1:50
        x1  = _assoc_Xstep(1,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x2  = _assoc_Xstep(2,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x3  = _assoc_Xstep(3,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x4  = _assoc_Xstep(4,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x5  = _assoc_Xstep(5,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x6  = _assoc_Xstep(6,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x7  = _assoc_Xstep(7,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x8  = _assoc_Xstep(8,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x9  = _assoc_Xstep(9,  X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x10 = _assoc_Xstep(10, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x11 = _assoc_Xstep(11, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x12 = _assoc_Xstep(12, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x13 = _assoc_Xstep(13, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x14 = _assoc_Xstep(14, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x15 = _assoc_Xstep(15, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x16 = _assoc_Xstep(16, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x17 = _assoc_Xstep(17, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x18 = _assoc_Xstep(18, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x19 = _assoc_Xstep(19, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        x20 = _assoc_Xstep(20, X, total, n0, xi, Δ_vals, params, n_pairs, Val(NC))
        X = (0.5*(X[1]+x1),  0.5*(X[2]+x2),  0.5*(X[3]+x3),  0.5*(X[4]+x4),
             0.5*(X[5]+x5),  0.5*(X[6]+x6),  0.5*(X[7]+x7),  0.5*(X[8]+x8),
             0.5*(X[9]+x9),  0.5*(X[10]+x10), 0.5*(X[11]+x11), 0.5*(X[12]+x12),
             0.5*(X[13]+x13), 0.5*(X[14]+x14), 0.5*(X[15]+x15), 0.5*(X[16]+x16),
             0.5*(X[17]+x17), 0.5*(X[18]+x18), 0.5*(X[19]+x19), 0.5*(X[20]+x20))
    end

    # Accumulate f_assoc = Σᵢ Σₐ n₀ᵢ ξᵢ n_ia (ln X_ia - X_ia/2 + 1/2)
    res = 0.0
    for i in 1:NC
        base = _nti(params.n_sites_cumsum, i)
        ns_i = _nti(params.n_sites_cumsum, i+1) - base
        n0i  = _nti(n0, i)
        xii  = _nti(xi, i)
        for a in 1:ns_i
            s        = base + a
            n_ia_val = _nti(params.n_sites_flat, s)
            X_val    = _nti(X, s)
            res += n0i * xii * n_ia_val *
                   (Base.log(abs(X_val)) - 0.5*X_val + 0.5)
        end
    end

    return res
end