struct CartesianDensityProfile{ℂ,ρ} <: DensityProfile{ℂ,ρ} #spline density profile. parametrize by dimensions?
    coords::ℂ
    density::ρ
    bounds::Vector{Float64}
    boundary_conditions::Vector{Float64}
    coeffs::Vector{NTuple{4,Float64}} #spline coefficients
    mesh_size::Float64
end


function CartesianDensityProfile(ρ,z,bounds,boundary_conditions)    
    coeffs = Vector{NTuple{4,Float64}}(undef,length(z)-1)
    mesh_size = (z[end]-z[1])/(length(z)-1)
    prof =  CartesianDensityProfile(z,ρ,bounds,boundary_conditions,coeffs,Float64(mesh_size))
    update_profile!(prof,prof.density)
end
