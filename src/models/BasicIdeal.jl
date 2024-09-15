using Clapeyron: BasicIdealModel

"""
    F_ideal(system::DFTSystem)

Obtain the ideal free energy of the system.

The output is a scalar of units J.
"""
function F_ideal(system::DFTSystem)
    model = system.model
    ρ = system.profiles
    ngrid = system.structure.ngrid

    dz = ρ[1].mesh_size

    n = zeros(ngrid,length(model))
    for i in @comps
        for k in @chain(i)
            n[:,i] += ρ[k].density/system.species.nbeads[i]
        end
    end
    
    f(x) = f_ideal(system,model.idealmodel,x)    

    ϕ = zeros(ngrid)
    
    Threads.@threads for i in 1:ngrid
        ϕ[i] = f(@view n[i,:])
    end

    return ∫(ϕ,dz)
end

function f_ideal(system::DFTSystem,model::BasicIdealModel,n)
    T = system.structure.conditions[2]
    ∑f = zero(T + first(n))
    lnT = log(T)
    return @sum(N_A*n[i]*(log(n[i]) - 1.5*lnT-1))
end