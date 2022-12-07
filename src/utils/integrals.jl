function ∫(f,dz)
    return sum(f)*dz
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