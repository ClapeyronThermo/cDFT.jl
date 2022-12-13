# function ∫(f,dz)
#     return sum(f)*dz
# end

function ∫(f,dz)
    return (f[1]+f[end]+2*sum(f[2:end-1]))*dz/2
end

# function ∫(f,dz)
#     return 1/3*dz*(f[1]+f[end]+4*sum(f[2:2:end-1])+2*sum(f[3:2:end-1]))
# end

function ∫fdz(f::Vector,z::Vector,z_eval::Float64,lim::Float64)
    dz = z[2]-z[1]
    N = Int(round(1/dz)*lim)
    id = findfirst(x->x==z_eval,z)
    idx = id-N:id+N
    return ∫(f[idx],dz)
end

function ∫fzdz(f::Vector,z::Vector,z_eval::Float64,lim::Float64)
    dz = z[2]-z[1]
    N = Int(round(1/dz)*lim)
    id = findfirst(x->x==z_eval,z)
    idx = id-N:id+N
    return ∫(f[idx].*(z[idx].-z_eval),dz)
end


function ∫fz²dz(f::Vector,z::Vector,z_eval::Float64,lim::Float64)
    dz = z[2]-z[1]
    N = Int(round(1/dz)*lim)
    id = findfirst(x->x==z_eval,z)
    idx = id-N:id+N
    return ∫(f[idx].*(lim^2 .-(z[idx].-z_eval).^2),dz)
end