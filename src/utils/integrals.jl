function ∫(f,dz)
    n = length(f)
    if n % 2 == 0 
        return (f[1] + f[n] + 4sum(f[2:2:n]) + 2sum(f[3:2:n-1]))*dz/3
    else
        return (f[1] + f[n-1] + 4sum(f[2:2:n-1]) + 2sum(f[3:2:n-2]))*dz/3 + f[n]*dz
    end
end

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