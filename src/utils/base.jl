"""
    free_energy(system::DFTSystem, ρ)

Obtain the total free energy of the system for a given profile. This is done by summing the ideal and residual free energies.

The output is a scalar of units J.
"""
function free_energy(system::Union{DFTSystem,DGTSystem}, ρ)
    return F_ideal(system,ρ)+F_res(system,ρ)
end

onevec(model) = Clapeyron.FillArrays.Ones(length(model))

macro chain(component, args...)
    quote
        if hasfield(typeof(model), :groups)
            model.groups.i_groups[$(component)]
        else
            $(component)
        end
    end |> esc
end

function get_chain_idx(model::EoSModel,i,j,a,b)
    return get_chain_idx(model.sites,i,j,a,b)
end

function get_chain_idx(param::SiteParam, i::Int64, j::Int64, a::Int64, b::Int64)
    if isnothing(param.site_translator)
        return i,j
    else
        site_translator::Vector{Vector{NTuple{2,Int}}} = param.site_translator
        k,_ = site_translator[i][a]
        l,_ = site_translator[j][b]
        return k,l
    end
end