import Clapeyron: a_res

include("BasicIdeal.jl")
include("DFT/dft.jl")
include("DGT/dgt.jl")

"""
    F_res(system::DFTSystem, ρ)

Obtain the residual free energy of the system for a given profile `ρ`. This is done by first evaluating the system fields and passing these to the integrands (`f_res`) for each grid point. The result is then integrated over the domain.

The output is a scalar of units J.
"""
function F_res(system::AbstractcDFTSystem, ρ)
    ngrid = system.structure.ngrid
    bounds = system.structure.bounds
    model = system.model

    _bounds = system.structure.bounds

    dz = structure_dz(system.structure)
    n = evaluate_field(system,ρ)

    f(x) = f_res(system,model,x)

    ϕ = similar(ρ,ngrid...)
    ϕ .= 0
    for kk in CartesianIndices(ngrid)
        k = Tuple(kk)
        ϕ[k...] = f(@view(n[k...,:,:]))
    end

    return ∫(ϕ,dz)
end

"""
    δFδρ_res(system::DFTSystem, ρ)

Obtain the functional derivatives of the residual free energy of the system for each component / bead for a given profile `ρ`. This is done by first evaluating the system fields, obtaining the derivative of the integrands (`f_res`) for each grid point and then integrating over each of these fields to obtain the functional derivatives.

The output is a 2D array with the dimensions `(ngrid,nb)`, where `ngrid` is the number of grid points, and `nb` is the number of beads in the model. The values are normalised by `kB*T`.
"""

function δFδρ_res!(system::AbstractcDFTSystem, ρ, δfδρ_res, n, δf, fft_buf, in_buf, out_buf, P, iP, f, cache_pool)
    model = system.model
    backend = system.options.device
    ngrid = system.structure.ngrid
    NF      = size(n, ndims(n)-1)
    NB      = size(n, ndims(n))
    ND      = length(ngrid)

    evaluate_field!(system, ρ, fft_buf, in_buf, out_buf, P, iP)

    copyto!(n, Adapt.adapt(typeof(n), fft_buf))

    Threads.@threads for kk in CartesianIndices(ngrid)
        k = Tuple(kk)
        cache = take!(cache_pool)
        ForwardDiff.gradient!(@view(δf[k...,:,:]), f, @view(n[k...,:,:]), cache)
        put!(cache_pool, cache)
    end

    copyto!(fft_buf, Adapt.adapt(typeof(fft_buf), δf))

    integrate_field!(system, fft_buf, δfδρ_res, in_buf, P, iP)
end

function length_scales(model::EoSModel)
    if hasfield(typeof(model.params), :sigma)
        return diagvalues(model.params.sigma.values)
    elseif hasfield(typeof(model.params), :b)
        return diagvalues(cbrt.(model.params.b.values/N_A))
    elseif hasfield(typeof(model.params), :lb_volume)
        return cbrt.(model.params.lb_volume.values/N_A)
    else
        error("No length scale defined in model")
    end
end