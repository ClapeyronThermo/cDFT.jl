using Clapeyron: HeterogcPCPSAFT, pcp_segment, pcp_sigma, pcp_epsilon, pcp_dipole2

function DFTSystem(model::HeterogcPCPSAFT,structure::DFTStructure,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure, options.device)
    propagator = get_propagator(model, species, structure, options.device)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, fields, nothing, propagator, options, chunksize)
end

function DFTSystem(model::HeterogcPCPSAFT,structure::DFTStructure, external_fields,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure, options.device)
    propagator = get_propagator(model, species, structure, options.device)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, fields, external_fields, propagator, options, chunksize)
end

struct gcPCPSAFTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::HeterogcPCPSAFT,structure::DFTStructure)
    (p,T) = structure.conditions
    z = structure.ρbulk
    v = 1/sum(z)

    HSd = d(model,1e-3,T,z)
    μres = Clapeyron.VT_chemical_potential_res(model, v, T, z/sum(z)) / Clapeyron.R̄ / T
    nbeads = length.(model.groups.groups)

    levels = zeros(Int, sum(nbeads))

    for i in @comps
        i_groups = model.groups.i_groups[i]
        bond_mat = Int.(model.groups.n_intergroups[i]) .> 0
        nbonds = sum(bond_mat,dims=2)[:]
        is_leaf = nbonds .== 1
        i_root = i_groups[findfirst(nbonds[i_groups] .== maximum(nbonds[i_groups]))]
        levels[i_root] = 1
    
        idx_current_level = i_root
        is_bonded = bond_mat[idx_current_level,:]
        k = 1
        while any(levels[i_groups] .== 0)
            levels[is_bonded] .= k+1
            idx_next_level = findall(levels .== k+1 .&& .!(is_leaf))
            is_bonded = (sum(bond_mat[idx_next_level,:],dims=1)[:].==1 .&& levels.==0)
            k+=1
        end
    end
    return gcPCPSAFTSpecies(nbeads,HSd,levels,structure.ρbulk,μres)
end

function get_fields(model::HeterogcPCPSAFT, species::DFTSpecies, structure::DFTStructure, backend::Backend)
    nb = sum(species.nbeads)
    ngrid = structure.ngrid
    ω = structure_ω(structure, backend)
    d = species.size
    ψ = 1.5357
    return [SWeightedDensity(:ρ,zeros(nb),ω,ngrid,backend),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,backend),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,backend),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,backend),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,backend),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,backend)]

end

function get_propagator(model::HeterogcPCPSAFT, species::DFTSpecies, structure::DFTStructure, backend::Backend)
    return TangentHSPropagator(model, species, structure, backend)
end

# function  Δ(model::HeterogcPCPSAFT, T, n, n₃, nᵥ)
#     ϵ_assoc = model.params.epsilon_assoc.values
#     κ = model.params.bondvol.values
#     σ = model.params.sigma.values
#     Δout = 
    
#     for (idx,(i,j),(a,b)) in indices(κ)
#         k,l = get_chain_idx(model,i,j,a,b)
#         gkl = @f(g_hs,k,l,_data)
#         Δout[idx] = gkl*σ[k,l]^3*(exp(ϵ_assoc[i,j][a,b]/T)-1)*κ[i,j][a,b]
#     end
#     return Δout
# end
function length_scale(model::HeterogcPCPSAFT)
    return maximum(model.params.sigma.values)
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for HeterogcPCPSAFT:
FMT hard-sphere + hard-chain bonding (Wertheim topology) + PCSAFT-style dispersion.
Polar term is NOT included in the Enzyme kernel (requires per-component aggregation).
Chain connectivity is handled by TangentHSPropagator.

Field layout (6 fields total):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc (bonding)
  5+ND     : ∫ρz²dz with d*ψ → ρ̄z  (dispersion, ψ=1.5357)

NC = total number of groups (sum of nbeads per component).
"""
@inline function f_res(::Type{M}, kk, out, n, params, T,
                       ::Val{NC}, ::Val{ND}) where {NC, ND, M <: HeterogcPCPSAFT}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc    = f_hc(M, kk, n, params, T, Val(NC), Val(ND))
    # res_disp  = f_disp(M, kk, n, params, T, Val(NC), Val(ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_hc + res_assoc
    return nothing
end

@inline function f_hc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: HeterogcPCPSAFT}
    _pi   = 3.141592653589793
    eps_v = 1e-15
    HSd   = params.HSd
    m_seg = params.m
    bond_k  = params.bond_k
    bond_l  = params.bond_l

    idx_ζ = 4 + ND
    ζ₃ = 0.0; ζ₂ = 0.0
    @inbounds for i in 1:NC
        mi = m_seg[i]; di = HSd[i]; ρ̄hci = n[kk, idx_ζ, i]
        ζ₃ += mi * ρ̄hci
        ζ₂ += mi * ρ̄hci / di
    end
    ζ₃ *= 0.125; ζ₂ *= 0.125
    inv1ζ₃ = 1.0 / (1.0 - ζ₃ + eps_v)

    return _f_hc_bonds(n, bond_k, bond_l, HSd, kk, ζ₂, inv1ζ₃)
end

# Bond-loop helper: NB is a WHERE-clause type param so the loop bound is always
# compile-time inside this function, regardless of how autodiff_deferred specialises.
@inline function _f_hc_bonds(n, bond_k::NTuple{NB, Int32}, bond_l::NTuple{NB, Int32},
                               HSd, kk, ζ₂, inv1ζ₃) where NB
    eps_v = 1e-15
    res_hc = 0.0
    @inbounds for ib in 1:NB
        k = _nti(bond_k, ib); l = _nti(bond_l, ib)
        dk = HSd[k]; dl = HSd[l]
        r_HSd = dk * dl / (dk + dl)
        ζ₂_ov3 = ζ₂ * inv1ζ₃
        yᵈᵈ = inv1ζ₃ + 3.0*r_HSd*ζ₂_ov3*inv1ζ₃ + 2.0*r_HSd^2*ζ₂_ov3^2*inv1ζ₃
        ρhck = n[kk, 1, k]
        res_hc -= ρhck * 0.5 * Base.log(abs(yᵈᵈ) + eps_v)
    end
    return res_hc
end

@inline function f_disp(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: HeterogcPCPSAFT}
    _pi   = 3.141592653589793
    eps_v = 1e-15
    HSd              = params.HSd
    m_seg            = params.m
    σ                = params.sigma
    ϵ                = params.epsilon
    nbeads_for_group = params.nbeads_for_group

    ψ       = 1.5357
    idx_ρz  = 5 + ND
    factor  = 3.0 / (4.0*ψ*ψ*ψ*_pi)
    ρ̄_tot   = eps_v; m̄_num = 0.0; η_sum = 0.0
    @inbounds for i in 1:NC
        di  = HSd[i]
        ρ̄i  = n[kk, idx_ρz, i] * factor / (di*di*di)
        m̄_num += m_seg[i] * ρ̄i
        η_sum  += m_seg[i] * ρ̄i * di*di*di
        ρ̄_tot  += ρ̄i / _nti(nbeads_for_group, i)
    end
    m̄  = m̄_num / ρ̄_tot
    ηd = _pi/6.0 * η_sum

    m2ϵσ3_1 = 0.0; m2ϵσ3_2 = 0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄i   = n[kk, idx_ρz, i] * factor / (di*di*di)
        @inbounds for j in 1:NC
            dj   = HSd[j]
            ρ̄j   = n[kk, idx_ρz, j] * factor / (dj*dj*dj)
            cij  = ρ̄i * ρ̄j * m_seg[i] * m_seg[j] * σ[i,j]*σ[i,j]*σ[i,j]
            eT   = ϵ[i,j] / (T + eps_v)
            m2ϵσ3_1 += cij * eT
            m2ϵσ3_2 += cij * eT * eT
        end
    end
    ηd2    = ηd*ηd
    ηd4    = (1.0-ηd+eps_v)^4
    inv1ηd = 1.0/(1.0-ηd+eps_v)
    inv2ηd = 1.0/(2.0-ηd+eps_v)
    C₁     = 1.0 + m̄*(8.0*ηd-2.0*ηd2)/ηd4 +
              (1.0-m̄)*(20.0*ηd-27.0*ηd2+12.0*(ηd*ηd2)-2.0*(ηd2*ηd2)) *
              inv1ηd*inv1ηd*inv2ηd*inv2ηd
    I₁     = I_lite(PCSAFT_CORR1, m̄, ηd)
    I₂     = I_lite(PCSAFT_CORR2, m̄, ηd)
    return -2.0*_pi*I₁*m2ϵσ3_1 - _pi*m̄*I₂*m2ϵσ3_2 / (C₁ + eps_v)
end

function preallocate_params(system::DFTSystem{<:HeterogcPCPSAFT})
    backend  = system.options.device
    model    = system.model
    nc_spec  = length(model)
    nbeads   = system.species.nbeads
    nc_groups = sum(nbeads)

    bond_k_list = Int32[]
    bond_l_list = Int32[]
    for i in 1:nc_spec
        i_groups        = model.groups.i_groups[i]
        n_intergroups_i = model.groups.n_intergroups[i]
        for k in i_groups
            for l in findall(n_intergroups_i[k,:] .== 1)
                push!(bond_k_list, Int32(k))
                push!(bond_l_list, Int32(l))
            end
        end
    end

    nbeads_for_group = Vector{Float64}(undef, nc_groups)
    for i in 1:nc_spec
        nbi = Float64(nbeads[i])
        for k in model.groups.i_groups[i]
            nbeads_for_group[k] = nbi
        end
    end

    n_bonds = length(bond_k_list)
    bond_k_t           = ntuple(ib -> Int32(bond_k_list[ib]), n_bonds)
    bond_l_t           = ntuple(ib -> Int32(bond_l_list[ib]), n_bonds)
    nbeads_for_group_t = ntuple(i  -> Float64(nbeads_for_group[i]), nc_groups)

    base = (;
        HSd              = Adapt.adapt(backend, system.species.size),
        m                = Adapt.adapt(backend, system.model.params.segment.values),
        sigma            = Adapt.adapt(backend, system.model.params.sigma.values),
        epsilon          = Adapt.adapt(backend, system.model.params.epsilon.values),
        nbeads_for_group = nbeads_for_group_t,
        bond_k           = bond_k_t,
        bond_l           = bond_l_t,
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, _ispec, _jspec,
         assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params_gc(model, system.species.size)

        assoc_icomp_t    = ntuple(p -> p <= nn           ? assoc_icomp_v[p]    : 0, Val(20))
        assoc_jcomp_t    = ntuple(p -> p <= nn           ? assoc_jcomp_v[p]    : 0, Val(20))
        assoc_isite_t    = ntuple(p -> p <= nn           ? assoc_isite_v[p]    : 0, Val(20))
        assoc_jsite_t    = ntuple(p -> p <= nn           ? assoc_jsite_v[p]    : 0, Val(20))
        n_sites_flat_t   = ntuple(j -> j <= total_sites  ? n_sites_flat_v[j]   : 0, Val(20))
        n_sites_cumsum_t = ntuple(i -> i <= nc_groups + 1
                                       ? n_sites_cumsum_v[i]
                                       : n_sites_cumsum_v[nc_groups + 1], Val(30))

        assoc = (;
            has_assoc      = true,
            assoc_n_pairs  = Val(nn),
            assoc_icomp    = assoc_icomp_t,
            assoc_jcomp    = assoc_jcomp_t,
            assoc_isite    = assoc_isite_t,
            assoc_jsite    = assoc_jsite_t,
            assoc_eps      = Adapt.adapt(backend, assoc_eps_v),
            assoc_kap      = Adapt.adapt(backend, assoc_kap_v),
            assoc_sig3     = Adapt.adapt(backend, assoc_sig3_v),
            assoc_dij      = Adapt.adapt(backend, assoc_dij_v),
            n_sites_flat   = n_sites_flat_t,
            n_sites_cumsum = n_sites_cumsum_t,
            total_sites,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    return params, nc_groups
end