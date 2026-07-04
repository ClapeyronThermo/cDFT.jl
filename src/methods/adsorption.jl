"""
    adsorption(system::AbstractcDFTSystem, ρ)

Calculate the (per-species) adsorption of a converged density profile `ρ` at a surface, defined as the excess amount of each species per unit volume of the accessible domain. To get accurate results, the system should be converged first.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> surface = Steele(["graphite"])

julia> adsorption(model, surface, 1e5, 298.15)
```
"""
function adsorption(system,ρ)
    # Integrate over all profiles
    nc = length(system.model)
    nd = dimension(system)
    if nd == 1
        V = prod(diff(system.structure.bounds))
    else
        V = prod(diff(system.structure.bounds; dims=2))
    end
    return [∫(selectdim(ρ,nd+1,i)[:],system.structure)/V for i in 1:nc]
end

"""
    adsorption(model::EoSModel, surface::ExternalFieldModel, p, T, n=[1.0])

Calculate the adsorption of a given `model` at conditions `p`, `T` and bulk composition `n`, next to an external field `surface` (e.g. a `Steele` wall). This is a blind calculation that assumes the domain is 1D and cartesian, sets up the system between the wall and the edge of `surface`'s `width`, converges the density profile, and then integrates it via the `AbstractcDFTSystem` method above.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> surface = Steele(["graphite"])

julia> adsorption(model, surface, 1e5, 298.15)
```
"""
function adsorption(model::EoSModel, surface::ExternalFieldModel, p, T, n=[1.0])
    L = cDFT.length_scale(model)

    width = surface.params.width[1]
    bounds = [0.7L,width-0.7L]

    v = volume(model,p,T,n)
    ρ = n./v

    structure = cDFT.Uniform1DCart((p, T), ρ, bounds, (201,))

    system = cDFT.DFTSystem(model, structure, surface)

    ρ = initialize_profiles(system)

    converge!(system,ρ)

    return adsorption(system, ρ)
end

export adsorption