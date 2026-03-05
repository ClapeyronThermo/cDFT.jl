tanh_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*tanh((x-shift)*coef)+1/2*(start+stop)
cos_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*cos((x-shift)*2π)*sqrt((1+coef^2)/(1+coef^2*cos((x-shift)*2π)^2))+1/2*(start+stop)

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
function initialize_profiles(system::AbstractcDFTSystem)
    ρ = initialize_profiles(system.model,system.structure,system.species,system.options.device)
    if system.external_field == nothing
        return ρ
    else
        for i in system.external_field
            if !(i isa ElectrostaticPotentialModel)
                 initialize_profiles!(system, i, ρ)
            end
        end
        return ρ
    end
end

function initialize_profiles(model::EoSModel,structure::Uniform1DCart, species, device)
    ngrid = structure.ngrid
    ρbulk = structure.ρbulk

    ρ = allocate(device, Float64, ngrid..., sum(species.nbeads))

    for i in @comps
        for j in @chain(i)
            ρ[:,j] = ρbulk[i]*ones(ngrid)
        end
    end
    return ρ
end

function get_coords(structure::DFTStructure)
    ngrid = structure.ngrid
    nd = length(ngrid)
    bounds = structure.bounds
    z = [uniform_range(structure,i) |> collect for i in 1:nd]
    Z = zeros(ngrid...,nd)
    for jj in CartesianIndices(ngrid)
        j = Tuple(jj)
        for i in 1:nd
            Z[j...,i] = z[i][j[i]]
        end
    end
    return Z
end

function structure_fftfreq(structure::DFTStructure)
    ngrid = structure.ngrid
    nd = dimension(structure)
    function ff(i)
        lb,ub = bounds(structure,i)
        return ngrid[i]/(ub - lb)
    end
    f = ntuple(ff,nd)
    ω = fftfreq.(ngrid, f)
end

function structure_ω(structure::DFTStructure)
    ngrid = structure.ngrid
    ω̂ = structure_fftfreq(structure)
    ω = zeros(ngrid...,length(ngrid))
    for kk in CartesianIndices(ngrid)
        k = Tuple(kk)
        for i in 1:length(ngrid)
            ω[k...,i] = ω̂[i][k[i]]
        end
    end
    ω
end

function structure_dz(structure::DFTStructure)
    ngrid = structure.ngrid
    nd = dimension(structure)
    function ff(i)
        lb,ub = bounds(structure,i)
        return (ub - lb)/ngrid[i]
    end
    return ntuple(ff,nd)
end