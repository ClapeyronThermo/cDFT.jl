"""
    interfacial_tension(model::EoSModel,p,T,n)

Calculate the interfacial tension of a liquid-liquid interface based on a given `model` and at conditions `p`, `T` and `n`. This is a blind calculation that assumes the interface is 1D and cartesian, and that the system does phase split between two bulk phases. If the system is expected to form micelles, other methods should be used.

Example:
```julia
julia> model = PCSAFT(["water","hexane"])

julia> interfacial_tension(model,1e5,298.15,[0.5,0.5])
```
"""
function interfacial_tension(model::EoSModel,p,T,n)
    L = length_scale(model)

    (x,n,G) = tp_flash(model, 1e5, 298.15, [0.5,0.5], RRTPFlash(equilibrium=:lle))

    structure = InterfacialTension1DCart((p, T, x[1,:]),[-10L,10L], 201, x[2,:])

    system = DFTSystem(model, structure)

    converge!(system)

    ρ = system.profiles
    ngrid = system.structure.ngrid

    F = free_energy(system)

    ρl =[ρ[i].boundary_conditions[2].value for i in @comps]
    x = ρl/sum(ρl)
    vl = 1/sum(ρl)

    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)

    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(ones(ngrid),ρ[1].mesh_size)
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
interfacial_tension(system::DFTSystem) = surface_tension(system)

export interfacial_tension