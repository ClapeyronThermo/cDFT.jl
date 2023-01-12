struct DensityProfile #spline density profile. parametrize by dimensions?
    coords::Vector{Float64}
    density::Vector{Float64}
    bounds::Vector{Float64}
    boundary_conditions::Vector{Float64}
    coeffs::Vector{NTuple{4,Float64}} #spline coefficients
    mesh_size::Float64
end


function DensityProfile(ρ,z,bounds,boundary_conditions)    
    coeffs = Vector{NTuple{4,Float64}}(undef,length(z)-1)
    mesh_size = (z[end]-z[1])/(length(z)-1)
    prof =  DensityProfile(z,ρ,bounds,boundary_conditions,coeffs,mesh_size)
    update_profile!(prof,prof.density)
end

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

    zs = vcat(za,zb,z,zy,zz)
    ρs = vcat(boundary_conditions[1],boundary_conditions[1],ρ,boundary_conditions[2],boundary_conditions[2])

    ms = (ρs[2:end].-ρs[1:end-1])./(zs[2:end].-zs[1:end-1])

    ts = zeros(eltype(ρs), length(zs))

    @inbounds for i in 3:length(zs)-2
        # Equals (1) in [1]
        m1, m2, m3, m4 = @view(ms[i-2:i+1])
        # As described in [1] p.591 (parentheses block), this is an arbitrary
        # convention to guarantee uniqueness of the solution
        if m1 == m2 && m3 == m4
            ts[i] = (m2+m3)/2
        else
            numer = abs(m4-m3)*m2 + abs(m2-m1)*m3
            denom = abs(m4-m3)    + abs(m2-m1)
            ts[i] = numer / denom
        end
    end

    for i in 1:(length(z)-1)
        x1, x2 = zs[i+2],zs[i+3]
        y1, y2 = ρs[i+2],ρs[i+3]
        t1, t2 = ts[i+2],ts[i+3]
        p0 = y1
        p1 = t1
        p2 = (3(y2-y1)/(x2-x1)-2t1-t2)/(x2-x1)
        p3 = (t1+t2-2(y2-y1)/(x2-x1))/(x2-x1)^2
        coeffs[i] = (p0, p1, p2, p3)
    end
    return prof
end

function (ρ::DensityProfile)(z)
    # Value sits on upper interval border
    @inbounds begin
    if z == ρ.coords[end]
        idx = length(ρ.coords)-1

    # Find the corresponding interval
    else

        # Find the interval index
        idx = _binary_search_interval(ρ.coords, z)

        # Extrapolation to the "left"
        if idx == 0
            return ρ.boundary_conditions[1]
        end

        # Extrapolation to the "right"
        if idx == length(ρ.coords)
            return ρ.boundary_conditions[2]
        end
    end

    # Evaluate the spline segment
    
    coeffs = ρ.coeffs[idx]
    res = evalpoly(z-ρ.coords[idx], coeffs)
    return res
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