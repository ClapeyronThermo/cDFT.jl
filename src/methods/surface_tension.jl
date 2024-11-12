"""
    surface_tension(model::EoSModel,T,x)

Calculate the surface tensions of a vapour-liquid interface based on a given `model` and at saturated conditions `T` and `x` (liquid composition). This is a blind calculation that assumes the interface is 1D and cartesian, and that the system does phase split between two bulk phases. If the system is expected to form micelles, other methods should be used.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> surface_tension(model,298.15,[1.0])
```
"""
function surface_tension(model::EoSModel, T, x = [1.0])
    L = length_scale(model)

    (p, vl, vv, y) = bubble_pressure(model, T, x)
    
    ρ1 = x./vl
    ρ2 = y./vv

    structure = TwoPhase1DCart((p, T), ρ1, ρ2, [-10L,10L], 201)

    system = DFTSystem(model, structure)

    ρ = initialize_profiles(system)

    ρ = converge!(system, ρ)
    return surface_tension(system,ρ)/2
end

function surface_tension(system::DFTSystem,ρ)
    model = system.model
    ngrid = system.structure.ngrid
    bounds = system.structure.bounds
    dz = (bounds[2]-bounds[1])/(ngrid)

    F = free_energy(system,ρ)

    (p, T) = system.structure.conditions
    n = zeros(length(model))
    
    ρl = system.structure.ρbulk
    x = ρl/sum(ρl)
    vl = 1/sum(ρl)

    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    chem_pot_term = 0.
    for i in @comps
        for k in @chain(i)
            chem_pot_term += μ[i]*∫(ρ[:,k],dz)/system.species.nbeads[i]
        end
    end
    return F*k_B*T-chem_pot_term+p*∫(ones(ngrid),dz)
end

export surface_tension