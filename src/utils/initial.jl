function initial_uniform_density_profile(model::EoSModel,ρ_val,bounds,ngrid::Int64=101)
    return _initial_uniform_density_profile(
        model,:Cartesian,ρ_val,bounds,ngrid
    )
end
    
    

function initial_uniform_spherical_density_profile(model::EoSModel,ρ_val,bounds,ngrid::Int64=101)
    return _initial_uniform_spherical_density_profile(
        model,:Spherical,ρ_val,bounds,ngrid
    )
end

function _initial_uniform_spherical_density_profile(model::EoSModel,prof_type::Symbol,ρ_val,bounds,ngrid::Int64=101)
    DP = prof_type == :Cartesian ? CartesianDensityProfile : SphericalDensityProfile
    
    z = range(first(bounds),last(bounds),ngrid) |> collect
    ρ = DP[]

    for i in @comps
        boundary_conditions = [ρ_val[i],ρ_val[i]]

        ρ_points = ρ_val[i]*ones(length(z))

        push!(ρ,DP(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end