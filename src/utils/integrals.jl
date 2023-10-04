
"""
    ∫(f,dz)

Integrates a collection of points `f`, with constant `dz`, using simpson rule

"""
∫(f,dz) = _∫(f,dz)
∫(f,dz,lastidx) = _∫(f,dz,lastidx)
#function _∫(f::AbstractArray,dz::Number,lastidx)
#    return 1/3*dz*(f[1]+f[end]+4*sum(@view(f[2:2:end-1]))+2*sum(@view(f[3:2:end-1])))
#end

function _∫(f,dz::Number,last = 0)
    ∑f = zero(typeof(dz))
    for (i,fi) in enumerate(f)
        if i == 1 || i == last
            ∑f += fi
        else
            ∑f += (4 - 2*(i % 2)) * fi #4 if is even, 2 if is odd
        end
    end
    return ∑f*dz/3
end

function ∫ρdz(ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)*one(eltype(ρ.density))
    z = z_eval.+span
    I = (ρ(zi) for zi in z)
    return ∫(I,dz,length(z))
end

function ∫ρzdz(ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)*one(eltype(ρ.density))
    z = z_eval.+span
    I = (ρ(zi) * (zi - z_eval) for zi in z)
    return  ∫(I,dz,length(z))
end

function ∫ρz²dz(ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)
    z = z_eval.+span
    I = (ρ(zi)*(span[end]^2 - (zi - z_eval)^2) for zi in z)
    return ∫(I,dz,length(z))
end
#=
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
end=#