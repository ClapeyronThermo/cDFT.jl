tanh_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*tanh((x-shift)*coef)+1/2*(start+stop)

include("surface_tension.jl")
include("interfacial_tension.jl")

"""
    initialize_profiles(model::EoSModel,structure::DFTStructure)

Based on the structure, this function will initialize the density profiles for each of the species / beads in the model. The output will be a vector of unconverged `DFTProfile`s.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> L = length_scale(model)

julia> structure = Uniform1DCart((1e5, 298.15, [1.]), [0, 20L], 201)

julia> profiles = initialize_profiles(model,structure)
```
"""
function initialize_profiles(model::EoSModel,structure::Uniform1DCart, species)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, x) = structure.conditions

    z = LinRange(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    vol = Clapeyron.volume(model,p,T,x)

    ρl = x./vol

    ρ = DensityProfile[]

    for i in @comps
        nbeads = species[i].nbeads
        for j in 1:nbeads
            boundary_conditions = (FixedBoundary(ρl[i],-1), FixedBoundary(ρl[i],1))
            ρ_points = ρl[i]*ones(ngrid)
            push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::Uniform1DSphr,species)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, x) = structure.conditions

    z = LinRange(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    vol = Clapeyron.volume(model,p,T,x)

    ρl = x./vol

    ρ = DensityProfile[]
    for i in @comps
        nbeads = species[i].nbeads
        for j in 1:nbeads
            boundary_conditions = (FreeBoundary(ρl[i],-1), FixedBoundary(ρl[i],1))
            ρ_points = ρl[i]*ones(ngrid)
            push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
        end
    end
    return ρ
end