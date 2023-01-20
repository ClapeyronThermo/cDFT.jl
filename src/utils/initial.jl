function initial_uniform_density_profile(model::EoSModel,ρ_val,bounds,ngrid::Int64=101)
    z = range(first(bounds),last(bounds),ngrid) |> collect

    ρ = DensityProfile[]

    for i in @comps
        boundary_conditions = [ρ_val[i],ρ_val[i]]

        ρ_points = ρ_val[i]*ones(length(z))

        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

