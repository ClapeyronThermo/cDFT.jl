struct SphericalDensityProfile{ℂ,ρ} <: DensityProfile{ℂ,ρ}
    coords::ℂ
    density::ρ
    bounds::Vector{Float64}
    boundary_conditions::Vector{Float64}
    coeffs::Vector{NTuple{4,Float64}} #spline coefficients
    mesh_size::Float64
end

function SphericalDensityProfile(ρ,z,bounds,boundary_conditions)
    if any(bounds.<0) || any(z.<0)
        error("SphericalDensityProfile: bounds must be non-negative.")
    elseif bounds[1] != 0. || z[1] != 0.
        error("SphericalDensityProfile: bounds must start at 0.")
    end
    coeffs = Vector{NTuple{4,Float64}}(undef,length(z)-1)
    mesh_size = (z[end]-z[1])/(length(z)-1)
    prof =  SphericalDensityProfile(z,ρ,bounds,boundary_conditions,coeffs,Float64(mesh_size))
    update_profile!(prof,prof.density)
end