"""
    F_res(system::DFTSystem, ρ)

Obtain the residual free energy of the system for a given profile `ρ`. This is done by first evaluating the system fields and passing these to the integrands (`f_res`) for each grid point. The result is then integrated over the domain.

The output is a scalar of units J.
"""
function F_res(system::DFTSystem, ρ)
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
function δFδρ_res(system::DFTSystem, ρ)
    model = system.model
    fields = system.fields
    
    nf = length(fields)
    ngrid = system.structure.ngrid
    nd = dimension(system)
    nb = size(ρ,nd+1)

    n = evaluate_field(system,ρ)
    nf = length_fields(system)
    # @assert nf == length(system.fields) "define length_fields(model::EoSModel) = nf"
    f(x) = f_res(system,model,x)
    idx_first = ntuple(Returns(1),nd)
    n_first = @view(n[idx_first...,:,:])
    cache = [ForwardDiff.GradientConfig(f,n_first, system.chunksize) for i in 1:Threads.nthreads()]
    df!(df,x) = ForwardDiff.gradient!(df,f,x,cache[Threads.threadid()])

    δf = zeros(ngrid...,nf,nb)
    Threads.@threads for kk in CartesianIndices(ngrid)
        k = Tuple(kk)
        df!(@view(δf[k...,:,:]),@view(n[k...,:,:]))
    end
    δFδρ_res = integrate_field(system, δf)
    return δFδρ_res
end

include("assoc.jl")