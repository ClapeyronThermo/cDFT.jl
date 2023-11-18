struct SphericalDensityProfile{ℂ,ρ} <: DensityProfile{ℂ,ρ}
    coords::ℂ
    density::ρ
    bounds::Vector{Float64}
    boundary_conditions::Vector{Float64}
    coeffs::Vector{NTuple{4,Float64}} #spline coefficients
    mesh_size::Float64
end

function SphericalDensityProfile(ρ,r,bounds,boundary_conditions)
    if any(bounds.<0)
        error("SphericalDensityProfile: bounds must be non-negative.")
    end
    coeffs = Vector{NTuple{4,Float64}}(undef,length(z)-1)
    mesh_size = (r[end]-r[1])/(length(r)-1)
    prof =  SphericalDensityProfile(r,ρ,bounds,boundary_conditions,coeffs,Float64(mesh_size))
    update_profile!(prof,prof.density)
end

# function update_profile(prof::SphericalDensityProfile,ρnew)
#     @assert length(prof.density) == length(ρnew)
#     prof.density .= ρnew
#     ρ = prof.density
#     r = prof.coords

#     bounds = prof.bounds
#     return prof
# end

# function (ρ::SphericalDensityProfile)(r)
#     # Value sits on upper interval border
#     @inbounds begin
#     if r == ρ.coords[end]
#         return ρ.density[end]
#     end
#     # Value sits on lower interval border
#     if r == ρ.coords[1]
#         return ρ.density[1]
#     end
#     # Value sits on lower interval border
#     if r < ρ.coords[1]
#         return ρ.boundary_conditions[1]
#     end
#     # Value sits on upper interval border
#     if r > ρ.coords[end]
#         return ρ.boundary_conditions[2]
#     end
#     # Value sits inside interval
#     i = searchsortedlast(ρ.coords,r)
#     x1, x2 = ρ.coords[i],ρ.coords[i+1]
#     y1, y2 = ρ.density[i],ρ.density[i+1]
#     t1, t2 = ρ.coeffs[i][1],ρ.coeffs[i][2]
#     t3, t4 = ρ.coeffs[i][3],ρ.coeffs[i][4]
#     return _spline_eval(r,x1,x2,y1,y2,t1,t2,t3,t4)
#     end
# end