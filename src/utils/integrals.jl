function ∫(f,dz)
    return sum(f)*dz
end

# function ∫(f,dz)
#     n = length(f)
#     return (3/8*(f[1] + f[n]) + 7/6*(f[2]+f[n-1]) + 23/24*(f[3]+f[n-2]) + sum(f[4:n-3]))*dz
# end

function ∫fdz(f::Vector,z::Vector,z_eval::Float64,lim::Float64)
    idx = @. z_eval-lim<=z && z<=z_eval+lim
    dz = z[2]-z[1]
    return ∫(f[idx[:]],dz)
end

function ∫fzdz(f::Vector,z::Vector,z_eval::Float64,lim::Float64)
    idx = @. z_eval-lim<=z && z<=z_eval+lim
    dz = z[2]-z[1]
    return ∫(f[idx[:]].*(z[idx[:]].-z_eval),dz)
end


function ∫fz²dz(f::Vector,z::Vector,z_eval::Float64,lim::Float64)
    idx = @. z_eval-lim<=z && z<=z_eval+lim
    dz = z[2]-z[1]
    return ∫(f[idx[:]].*(lim^2 .-(z[idx[:]].-z_eval).^2),dz)
end