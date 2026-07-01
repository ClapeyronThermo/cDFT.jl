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
    FP       = fptype(system.options)
    ppcmodel = system.model.ppcmodel
    dd_a_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_a[i][j]), 3), 5)
    dd_b_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_b[i][j]), 3), 5)
    dd_c_fp = ntuple(i -> ntuple(j -> FP(DD_consts.corr_c[i][j]), 3), 5)
    base = (;
        HSd         = adapt_to_device(backend, FP, system.species.size),
        m           = adapt_to_device(backend, FP, ppcmodel.params.segment.values),
        sigma       = adapt_to_device(backend, FP, ppcmodel.params.sigma.values),
        epsilon     = adapt_to_device(backend, FP, ppcmodel.params.epsilon.values),
        pcp_m       = adapt_to_device(backend, FP, pcp_segment(ppcmodel)),
        pcp_sigma   = adapt_to_device(backend, FP, pcp_sigma(ppcmodel)),
        pcp_epsilon = adapt_to_device(backend, FP, pcp_epsilon(ppcmodel)),
        dipole2     = adapt_to_device(backend, FP, pcp_dipole2(ppcmodel)),
        dd_a        = dd_a_fp,
        dd_b        = dd_b_fp,
        dd_c        = dd_c_fp,
    )

    nn = Clapeyron.assoc_pair_length(ppcmodel)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params(ppcmodel, system.species.size)

        nc_model         = length(ppcmodel)
        ia_global_v      = [n_sites_cumsum_v[assoc_icomp_v[p]] + assoc_isite_v[p] for p in 1:nn]
        jb_global_v      = [n_sites_cumsum_v[assoc_jcomp_v[p]] + assoc_jsite_v[p] for p in 1:nn]
        n_ia_v           = [n_sites_flat_v[ia_global_v[p]] for p in 1:nn]
        n_jb_v           = [n_sites_flat_v[jb_global_v[p]] for p in 1:nn]
        assoc_icomp_t    = ntuple(p -> assoc_icomp_v[p],    Val(nn))
        assoc_jcomp_t    = ntuple(p -> assoc_jcomp_v[p],    Val(nn))
        assoc_isite_t    = ntuple(p -> assoc_isite_v[p],    Val(nn))
        assoc_jsite_t    = ntuple(p -> assoc_jsite_v[p],    Val(nn))
        assoc_ia_global_t = ntuple(p -> ia_global_v[p],     Val(nn))
        assoc_jb_global_t = ntuple(p -> jb_global_v[p],     Val(nn))
        assoc_n_ia_t      = ntuple(p -> n_ia_v[p],          Val(nn))
        assoc_n_jb_t      = ntuple(p -> n_jb_v[p],          Val(nn))
        n_sites_flat_t   = ntuple(j -> n_sites_flat_v[j],   Val(total_sites))
        n_sites_cumsum_t = ntuple(i -> n_sites_cumsum_v[i], Val(nc_model + 1))

        assoc = (;
            has_assoc       = true,
            assoc_n_pairs   = Val(nn),
            assoc_n_sites   = Val(total_sites),
            assoc_icomp     = assoc_icomp_t,
            assoc_jcomp     = assoc_jcomp_t,
            assoc_isite     = assoc_isite_t,
            assoc_jsite     = assoc_jsite_t,
            assoc_ia_global = assoc_ia_global_t,
            assoc_jb_global = assoc_jb_global_t,
            assoc_n_ia      = assoc_n_ia_t,
            assoc_n_jb      = assoc_n_jb_t,
            assoc_eps       = adapt_to_device(backend, FP, assoc_eps_v),
            assoc_kap       = adapt_to_device(backend, FP, assoc_kap_v),
            assoc_sig3      = adapt_to_device(backend, FP, assoc_sig3_v),
            assoc_dij       = adapt_to_device(backend, FP, assoc_dij_v),
            n_sites_flat    = n_sites_flat_t,
            n_sites_cumsum  = n_sites_cumsum_t,
            total_sites,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    return params, length(ppcmodel)
end