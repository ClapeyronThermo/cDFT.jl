"""
    F_res(system::DFTSystem)

Obtain the residual free energy of the system. This is done by first evaluating the system fields and passing these to the integrands (`f_res`) for each grid point. The result is then integrated over the domain.

The output is a scalar of units J.
"""
function F_res(system::DFTSystem)
    model = system.model
    dz = system.profiles[1].mesh_size
    ngrid = system.structure.ngrid

    n = evaluate_field(system)

    f(x) = f_res(system,model,x)

    ϕ = zeros(ngrid)

    Threads.@threads for i in 1:ngrid
        ϕ[i] = f(@view n[i,:,:])
    end

    return ∫(ϕ,dz)
end

"""
    δFδρ_res(system::DFTSystem)

Obtain the functional derivatives of the residual free energy of the system for each component / bead. This is done by first evaluating the system fields, obtaining the derivative of the integrands (`f_res`) for each grid point and then integrating over each of these fields to obtain the functional derivatives.

The output is a 2D array with the dimensions `(ngrid,nc)`, where `ngrid` is the number of grid points, and `nc` is the number of components in the model. The values are normalised by `kB*T`.
"""
function δFδρ_res(system::DFTSystem)
    model = system.model
    fields = system.fields
    ρ = system.profiles
    z = ρ[1].coords
    nb = length(ρ)
    nf = length(fields)
    dz = system.profiles[1].mesh_size
    ngrid = system.structure.ngrid

    n = evaluate_field(system)

    f(x) = f_res(system,model,x)
    df(x) = ForwardDiff.gradient(f,x)

    δf = zeros(ngrid,nf,nb)

    Threads.@threads for i in 1:ngrid
        δf[i,:,:] = df(n[i,:,:])
    end

    δFδρ_res = integrate_field(system, δf)

    return δFδρ_res
end

include("assoc.jl")