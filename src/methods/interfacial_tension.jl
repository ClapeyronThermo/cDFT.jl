"""
    interfacial_tension(model::EoSModel,p,T,n)

Calculate the interfacial tension of a liquid-liquid interface based on a given `model` and at conditions `p`, `T` and `n`. This is a blind calculation that assumes the interface is 1D and cartesian, and that the system does phase split between two bulk phases. If the system is expected to form micelles, other methods should be used.

Example:
```julia
julia> model = PCSAFT(["water","hexane"])

julia> interfacial_tension(model,1e5,298.15,[0.5,0.5])
```
"""
function interfacial_tension(model::EoSModel,p,T,x,xx)
    L = length_scale(model)

    v1 = volume(model,p,T,x)
    v2 = volume(model,p,T,xx)

    ρ1 = x./v1
    ρ2 = xx./v2

    structure = TwoPhase1DCart((p, T), ρ1, ρ2,[-20L,20L], 201)

    system = DFTSystem(model, structure)
    ρ = initialize_profiles(system)

    ρ = converge!(system,ρ)
    return interfacial_tension(system,ρ)/2
end

"""
    interfacial_tension(system::DFTSystem)

Calculate the interfacial tension of an based on a given `system`. To get accurate results, the system should be converged first. 

Example:
```julia
julia> model = PCSAFT(["water","hexane"])

julia> structure = InterfacialTension1DCart((1e5, 298.15, [0.5,0.5]), [-10L, 10L], 201)

julia> system = DFTSystem(model, structure)

julia> converge!(system)

julia> interfacial_tension(system)
```
"""
interfacial_tension(system::DFTSystem, ρ) = surface_tension(system, ρ)

export interfacial_tension