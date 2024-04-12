
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

function ∫ρdz(structure::DFTStructure1DCart,ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    I = 0.
    
    z1 = z_eval-span[end]
    z2 = z_eval+span[end]

    idx1 = _binary_search_interval(ρ.coords, z1)
    idx2 = _binary_search_interval(ρ.coords, z2)

    if idx1 == 0
        I += ρ.boundary_conditions[1]*(ρ.coords[1]-z1)
    else
        I += evalpoly(ρ.coords[idx1+1]-ρ.coords[idx1],(0.0,(ρ.coeffs[idx1]./(1,2,3,4))...))
        I -= evalpoly(z1-ρ.coords[idx1],(0.0,(ρ.coeffs[idx1]./(1,2,3,4))...))
    end

    if idx2 == structure.ngrid
        I += ρ.boundary_conditions[2]*(z2-ρ.coords[end])
    else
        I += evalpoly(z2-ρ.coords[idx2],(0.0,(ρ.coeffs[idx2]./(1,2,3,4))...))
    end

    for i in idx1+1:idx2-1
        I += evalpoly(ρ.coords[i+1]-ρ.coords[i],(0.0,(ρ.coeffs[i]./(1,2,3,4))...))
    end

    return I

    # dz = step(span)*one(eltype(ρ.density))
    # z = z_eval.+span
    # I = (ρ(zi) for zi in z)
    # return ∫(I,dz,length(z))
end

function ∫ρzdz(structure::DFTStructure1DCart,ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)*one(eltype(ρ.density))
    z = z_eval.+span
    I = (ρ(zi) * (zi - z_eval) for zi in z)
    return  ∫(I,dz,length(z))
end

function ∫ρz²dz(structure::DFTStructure1DCart,ρ::DensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)
    z = z_eval.+span
    I = (ρ(zi)*(span[end]^2 - (zi - z_eval)^2) for zi in z)
    return ∫(I,dz,length(z))*π
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