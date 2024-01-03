function update_profile!(prof::DensityProfile,ρnew)
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

    zs = vcat(za,zb,z,zy,zz) #z + 4
    ρs = vcat(boundary_conditions[1],boundary_conditions[1],ρ,boundary_conditions[2],boundary_conditions[2])

    _m(x,y) = (y[2:end].-y[1:end-1])./(x[2:end].-x[1:end-1])
    ms = _m(zs,ρs)
    ρs_start = (boundary_conditions[1],boundary_conditions[1],ρ[1],ρ[2])
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


# shifts the profile by z_shift to the right
# put -z_gds to shift the profile so that z_gds places at the center
function shift_profile!(prof::DensityProfile,z_shift)
    coords = prof.coords
    len = length(coords)
    new_coords = coords .- z_shift
    ρ_l = prof.boundary_conditions[1]
    ρ_r = prof.boundary_conditions[2]
    z_max = coords[end]
    z_min = coords[1]
    ρ_new = zeros(len)
    for i in 1:len
        if new_coords[i] < z_min
            ρ_new[i] = ρ_l
        elseif new_coords[i] > z_max
            ρ_new[i] = ρ_r
        else
            ρ_new[i] = prof(new_coords[i])
        end
    end
    update_profile!(prof,ρ_new)
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