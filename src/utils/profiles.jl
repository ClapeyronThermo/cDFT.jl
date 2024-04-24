"""
    DensityProfile(ρ,z,bounds,boundary_conditions)  

A spline density profile. This profile is parametrized by a set splines obtained from densities `ρ` at coordinates `z`, with a set of `bounds` which have specified `boundary_conditions`. The spline coefficients are calculated and stored in the profile. The struct carries the following information:
- `coords`: The coordinates of the density profile.
- `density`: The density values at the coordinates.
- `bounds`: The bounds of the density profile.
- `boundary_conditions`: The boundary conditions of the density profile.
- `coeffs`: The spline coefficients of the density profile.
- `mesh_size`: The mesh size of the density profile.
Once the profile is created, the output can be treated like a function, where the value at a given coordinate is obtained by evaluating the spline segment at that coordinate.

Example:
```julia
julia> z = LinRange(-1,1,10)

julia> ρ = @. exp(-z^2)

julia> bounds = [-1,1]

julia> boundary_conditions = (FixedBoundary(exp(-1),-1),FixedBoundary(exp(-1),1))

julia> profile = DensityProfile(ρ,z,bounds,boundary_conditions)

julia> profile(0.5)
"""
struct DensityProfile{ℂ,ρ} <: DFTProfile #spline density profile. parametrize by dimensions?
    coords::ℂ
    density::ρ
    bounds::Vector{Float64}
    boundary_conditions::Tuple{BoundaryCondition,BoundaryCondition}
    coeffs::Vector{NTuple{4,Float64}} #spline coefficients
    mesh_size::Float64
end

function DensityProfile(ρ,z,bounds,boundary_conditions)    
    coeffs = Vector{NTuple{4,Float64}}(undef,length(z)-1)
    mesh_size = (z[end]-z[1])/(length(z)-1)
    prof =  DensityProfile(z,ρ,bounds,boundary_conditions,coeffs,Float64(mesh_size))
    update_profile!(prof,prof.density)
end

"""
    update_profile!(prof,ρnew)

This function will update the density profile `prof` with the new density values `ρnew`. The spline coefficients will be recalculated and stored in the profile.
"""
function update_profile!(prof,ρnew)
    @assert length(prof.density) == length(ρnew)
    prof.density .= ρnew
    ρ = prof.density
    z = prof.coords

    bounds = prof.bounds
    boundary_conditions = prof.boundary_conditions
    coeffs = prof.coeffs

    dz = prof.mesh_size

    zb,za = z[1]-dz,z[1]-2*dz
    zy,zz = z[end]+dz,z[end]+2*dz

    bc = get_boundary_conditions!(boundary_conditions,ρ)

    zs = vcat(za,zb,z,zy,zz) #z + 4
    ρs = vcat(bc[1,1],bc[1,2],ρ,bc[2,1],bc[2,2])

    _m(x,y) = (y[2:end].-y[1:end-1])./(x[2:end].-x[1:end-1])
    ms = _m(zs,ρs)
    ρs_start = (bc[1,1],bc[1,2],ρ[1],ρ[2])
    zs_start = (za,zb,z[1],z[2])
    ms_start = _m(zs_start,ρs_start)
    ts = zeros(eltype(ρs), length(zs))

    @inbounds for i in 3:length(zs)-2
        # Equals (1) in [1]
        m1, m2, m3, m4 = @view(ms[i-2:i+1])
        ts[i] = _spline_ts(m1,m2,m3,m4)
    end
    
    for i in 1:(length(z)-1)
        x1, x2 = z[i],z[i+1]
        y1, y2 = ρ[i],ρ[i+1]
        t1, t2 = ts[i+2],ts[i+3] #first element: t[1], #last: ts(end+2)
        coeffs[i] = _spline_coeff_calc(x1,x2,y1,y2,t1,t2)
    end
    return prof
end

function (ρ::DensityProfile)(z)
    # Value sits on upper interval border
    @inbounds begin
    if z < ρ.coords[1] && ρ.boundary_conditions[1] isa PeriodicBoundary
        width = ρ.coords[end] - ρ.coords[1]
        return ρ(ρ.coords[end] + mod(z-ρ.coords[1], width))
    elseif z > ρ.coords[end] && ρ.boundary_conditions[2] isa PeriodicBoundary
        width = ρ.coords[end] - ρ.coords[1]
        return ρ(ρ.coords[1] + mod(z-ρ.coords[end], width))
    else

        # Find the interval index
        idx = _binary_search_interval(ρ.coords, z)

        # Extrapolation to the "left"
        if idx == 0
            return ρ.boundary_conditions[1].value
        end

        # Extrapolation to the "right"
        if idx == length(ρ.coords)
            return ρ.boundary_conditions[2].value
        end
    end

    # Evaluate the spline segment
    
    coeffs = ρ.coeffs[idx]
    res = evalpoly(z-ρ.coords[idx], coeffs)
    return res
    end
end

@inline function _binary_search_interval(ρ::DensityProfile, number::Real)

    @inbounds begin
        if number < ρ.coords[1] && ρ.boundary_conditions[1] isa PeriodicBoundary
            width = ρ.coords[end] - ρ.coords[1]
            return _binary_search_interval(ρ.coords,ρ.coords[end] + mod(number-ρ.coords[1], width))
        elseif number > ρ.coords[end] && ρ.boundary_conditions[2] isa PeriodicBoundary
            width = ρ.coords[end] - ρ.coords[1]
            return _binary_search_interval(ρ.coords,ρ.coords[end] + mod(number-ρ.coords[1], width))
        else
            return _binary_search_interval(ρ.coords, number)
        end
    end
end

@inline function _binary_search_interval(array::Vector{T}, number::Real) where T <: Real

    left = 1
    right = length(array)

    @inbounds while left <= right
        center = (left+right)÷2
        if array[center] > number
            right = center - 1
        else # array[center] <= number
            left = center + 1
        end
    end
    return left - 1
end

function _spline_ts(m1,m2,m3,m4)
    # As described in [1] p.591 (parentheses block), this is an arbitrary
    # convention to guarantee uniqueness of the solution
    if m1 == m2 && m3 == m4
        return (m2+m3)/2
    else
        numer = abs(m4-m3)*m2 + abs(m2-m1)*m3
        denom = abs(m4-m3)    + abs(m2-m1)
        return numer / denom
    end
end

function _spline_coeff_calc(x1,x2,y1,y2,t1,t2)
    p0 = y1
    p1 = t1
    Δx = x2-x1
    Δy = y2-y1
    p2 = (3*Δy/Δx-2*t1-t2)/Δx
    p3 = (t1+t2-2*Δy/Δx)/Δx^2
    return (Float64(p0), Float64(p1), Float64(p2), Float64(p3))
end