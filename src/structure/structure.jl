tanh_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*tanh((x-shift)*coef)+1/2*(start+stop)

# include("surface_tension.jl")
# include("interfacial_tension.jl")
include("two_phase.jl")
include("external_field.jl")

"""
    initialize_profiles(system::DFTSystem)

Based on the system specifications, this function will initialize the density profiles for each of the species / beads in the model. The output will be an array of size `(ngrid,nb)` where `ngrid` is the number of grid points used and `nb` is the number of beads.

Example:
```julia
julia> model = PCSAFT(["water"])

julia> ρbulk = [molar_density(model,1e5,298.15)]

julia> L = length_scale(model)

julia> structure = Uniform1DCart((1e5, 298.15), ρbulk, [0, 20L], 201)

julia> system = DFTSystem(model, structure)

julia> profiles = initialize_profiles(system)
```
"""
function initialize_profiles(system::DFTSystem)
    return initialize_profiles(system.model,system.structure,system.species)
end

function initialize_profiles(model::EoSModel,structure::Uniform1DCart, species)
    ngrid = structure.ngrid
    ρbulk = structure.ρbulk

    ρ = zeros(ngrid,sum(species.nbeads))

    for i in @comps
        for j in @chain(i)
            ρ[:,j] = ρbulk[i]*ones(ngrid)
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
        nbeads = species.nbeads[i]
        for j in 1:nbeads
            boundary_conditions = (FreeBoundary(ρl[i],-1), FixedBoundary(ρl[i],1))
            ρ_points = ρl[i]*ones(ngrid)
            push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
        end
    end
    return ρ
end