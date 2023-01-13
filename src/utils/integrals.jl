function ∫(f,dz)
    return 1/3*dz*(f[1]+f[end]+4*sum(@view(f[2:2:end-1]))+2*sum(@view(f[3:2:end-1])))
end

function ∫ρdz(ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)
    z = z_eval.+span
    I = ρ.(z)
    return ∫(I,dz)
end

function ∫ρzdz(ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)
    z = z_eval.+span
    I = ρ.(z)
    return ∫(I.*(z.-z_eval),dz)
end


function ∫ρz²dz(ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)
    z = z_eval.+span
    I = ρ.(z)
    return ∫(I.*(span[end]^2 .-(z.-z_eval).^2),dz)
end

function ∫fdz(model::FunctionalModel,f::Vector,z::Vector{Float64},z_eval::Float64,lim::Float64)
    ϵerr = sqrt(eps(lim))
    idx = @. z_eval-lim-ϵerr<=z && z<=z_eval+lim+ϵerr
    dz = model.domain.mesh_size

    return ∫(f[idx[:]],dz)
end

function ∫fzdz(model::FunctionalModel,f::Vector,z::Vector{Float64},z_eval::Float64,lim::Float64)
    ϵerr = sqrt(eps(lim))
    idx = @. z_eval-lim-ϵerr<=z && z<=z_eval+lim+ϵerr
    dz = model.domain.mesh_size

    return ∫(f[vec(idx)].*(z[vec(idx)].-z_eval),dz)
end


function ∫fz²dz(model::FunctionalModel,f::Vector,z::Vector{Float64},z_eval::Float64,lim::Float64)
    ϵerr = sqrt(eps(lim))
    idx = @. z_eval-lim-ϵerr<=z && z<=z_eval+lim+ϵerr
    dz = model.domain.mesh_size

    return ∫(f[idx[:]].*(lim^2 .-(z[idx[:]].-z_eval).^2),dz)
end