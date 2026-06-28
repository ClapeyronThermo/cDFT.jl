using Clapeyron: HomogcPCPSAFTModel, pcp_segment, pcp_sigma, pcp_epsilon, pcp_dipole2

function get_species(model::HomogcPCPSAFTModel,structure::DFTStructure)
    return get_species(model.ppcmodel,structure)
end

function length_scale(model::HomogcPCPSAFTModel)
    return length_scale(model.ppcmodel)
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Route HomogcPCPSAFT kernel calls to the PCPSAFTModel implementation.
The params NamedTuple is assembled from model.ppcmodel in preallocate_model,
so the field layout and parameter names are identical to PCPSAFTModel.
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: HomogcPCPSAFTModel}
    return f_res(PCPSAFTModel, kk, out, n, params, T, Val{NC}(), Val{ND}())
end

function preallocate_params(system::DFTSystem{<:HomogcPCPSAFTModel})
    backend  = system.options.device
    ppcmodel = system.model.ppcmodel
    base = (;
        HSd         = Adapt.adapt(backend, system.species.size),
        m           = Adapt.adapt(backend, ppcmodel.params.segment.values),
        sigma       = Adapt.adapt(backend, ppcmodel.params.sigma.values),
        epsilon     = Adapt.adapt(backend, ppcmodel.params.epsilon.values),
        pcp_m       = Adapt.adapt(backend, pcp_segment(ppcmodel)),
        pcp_sigma   = Adapt.adapt(backend, pcp_sigma(ppcmodel)),
        pcp_epsilon = Adapt.adapt(backend, pcp_epsilon(ppcmodel)),
        dipole2     = Adapt.adapt(backend, pcp_dipole2(ppcmodel)),
    )

    nn = Clapeyron.assoc_pair_length(ppcmodel)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params(ppcmodel, system.species.size)

        nc_model = length(ppcmodel)
        assoc_icomp_t    = ntuple(p -> p <= nn           ? assoc_icomp_v[p]    : 0, Val(20))
        assoc_jcomp_t    = ntuple(p -> p <= nn           ? assoc_jcomp_v[p]    : 0, Val(20))
        assoc_isite_t    = ntuple(p -> p <= nn           ? assoc_isite_v[p]    : 0, Val(20))
        assoc_jsite_t    = ntuple(p -> p <= nn           ? assoc_jsite_v[p]    : 0, Val(20))
        n_sites_flat_t   = ntuple(j -> j <= total_sites  ? n_sites_flat_v[j]   : 0, Val(20))
        n_sites_cumsum_t = ntuple(i -> i <= nc_model + 1 ? n_sites_cumsum_v[i] : 0, Val(11))

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

    return params, length(ppcmodel)
end