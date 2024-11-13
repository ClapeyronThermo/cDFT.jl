"""
    F_res(system::DFTSystem, ρ)

Obtain the residual free energy of the system for a given profile `ρ`. This is done by first evaluating the system fields and passing these to the integrands (`f_res`) for each grid point. The result is then integrated over the domain.

The output is a scalar of units J.
"""
function F_res(system::DFTSystem, ρ)
    ngrid = system.structure.ngrid
    bounds = system.structure.bounds
    model = system.model
    dz = (bounds[2]-bounds[1])/ngrid
    ngrid = system.structure.ngrid

    n = evaluate_field(system,ρ)

    f(x) = f_res(system,model,x)

    ϕ = similar(ρ,ngrid)
    ϕ .= 0
    for i in 1:ngrid
        ϕ[i] = f(@view n[i,:,:])
    end

    return ∫(ϕ,dz)
end

f_res(system,model,ρ,dcache) = f_res(system,model,ρ)

"""
    δFδρ_res(system::DFTSystem, ρ)

Obtain the functional derivatives of the residual free energy of the system for each component / bead for a given profile `ρ`. This is done by first evaluating the system fields, obtaining the derivative of the integrands (`f_res`) for each grid point and then integrating over each of these fields to obtain the functional derivatives.

The output is a 2D array with the dimensions `(ngrid,nb)`, where `ngrid` is the number of grid points, and `nb` is the number of beads in the model. The values are normalised by `kB*T`.
"""
function δFδρ_res(system::DFTSystem, ρ)
    model = system.model
    fields = system.fields
    nb = size(ρ,2)
    nf = length(fields)
    ngrid = system.structure.ngrid

    n = evaluate_field(system,ρ)
    nf = length_fields(system)
    # @assert nf == length(system.fields) "define length_fields(model::EoSModel) = nf"
    f(x) = f_res(system,model,x)
    n_first = @view(n[1,:,:])
    cfg = ForwardDiff.GradientConfig(f, n_first, ForwardDiff.Chunk{nf}())
    df!(df,x) = ForwardDiff.gradient!(df,f,x,cfg)

    δf = zeros(ngrid...,nf,nb)

    for k in Iterators.product([1:ngrid[i] for i in 1:length(ngrid)]...)
        df!(@view(δf[k...,:,:]),@view(n[k...,:,:]))
    end
    δFδρ_res = integrate_field(system, δf, ρ)
    return δFδρ_res
end

include("assoc.jl")