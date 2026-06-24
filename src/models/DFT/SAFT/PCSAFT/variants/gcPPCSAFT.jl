using Clapeyron: HomogcPCPSAFTModel, pcp_segment, pcp_sigma, pcp_epsilon, pcp_dipole2

function get_species(model::HomogcPCPSAFTModel,structure::DFTStructure)
    return get_species(model.ppcmodel,structure)
end

function f_res(system::DFTSystem, model::HomogcPCPSAFTModel,n)
    return f_res(system,model.ppcmodel,n)
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
@inline function f_res(out, n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: HomogcPCPSAFTModel}
    return f_res(out, n, params, T, kk, Val{NC}(), Val{ND}(), PCPSAFTModel)
end

function preallocate_params(system::DFTSystem{<:HomogcPCPSAFTModel})
    backend  = system.options.device
    ppcmodel = system.model.ppcmodel
    params = (;
        HSd         = Adapt.adapt(backend, system.species.size),
        m           = Adapt.adapt(backend, ppcmodel.params.segment.values),
        sigma       = Adapt.adapt(backend, ppcmodel.params.sigma.values),
        epsilon     = Adapt.adapt(backend, ppcmodel.params.epsilon.values),
        pcp_m       = Adapt.adapt(backend, pcp_segment(ppcmodel)),
        pcp_sigma   = Adapt.adapt(backend, pcp_sigma(ppcmodel)),
        pcp_epsilon = Adapt.adapt(backend, pcp_epsilon(ppcmodel)),
        dipole2     = Adapt.adapt(backend, pcp_dipole2(ppcmodel)),
    )
    nc = length(ppcmodel)
    return params, nc
end