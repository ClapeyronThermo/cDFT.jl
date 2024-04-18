include("surface_tension.jl")
include("interfacial_tension.jl")

function initialize_profiles(model::EoSModel,structure::Uniform1DCart)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, x) = structure.conditions

    z = LinRange(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    vol = Clapeyron.volume(model,p,T,x)

    ρl = x./vol

    ρ = DensityProfile[]
    for i in @comps
        boundary_conditions = [ρl[i],ρl[i]]
        ρ_points = ρl[i]*ones(ngrid)
        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::Uniform1DSphr)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, x) = structure.conditions

    z = LinRange(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    vol = Clapeyron.volume(model,p,T,x)

    ρl = x./vol

    ρ = DensityProfile[]
    for i in @comps
        boundary_conditions = [ρl[i],ρl[i]]
        ρ_points = ρl[i]*ones(ngrid)
        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ
end