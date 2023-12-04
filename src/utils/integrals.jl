
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

function ∫ρdz(ρ::CartesianDensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)*one(eltype(ρ.density))
    z = z_eval.+span
    I = (ρ(zi) for zi in z)
    return ∫(I,dz,length(z))
end

function ∫ρzdz(ρ::CartesianDensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)*one(eltype(ρ.density))
    z = z_eval.+span
    I = (ρ(zi) * (zi - z_eval) for zi in z)
    return  ∫(I,dz,length(z))
end

function ∫ρz²dz(ρ::CartesianDensityProfile,z_eval::Float64,span::StepRangeLen)
    dz = step(span)
    z = z_eval.+span
    I = (ρ(zi)*(span[end]^2 - (zi - z_eval)^2) for zi in z)
    return ∫(I,dz,length(z))
end

##### Spherical Density Profiles
## 10.1063/1.1520530
## Yu and Wu, J. Chem. Phys. 117, 10156 (2002)

function ∫ρdz(ρ::SphericalDensityProfile,z_eval::Float64,span::StepRangeLen)
    return 1. /z_eval * ∫ρrdr(ρ,z_eval,span)
end
function ∫ρzdz(ρ::SphericalDensityProfile,z_eval::Float64,span::StepRangeLen)
    return 1. /(z_eval*z_eval) * ∫ρrr²dr_vector(ρ,z_eval,span)
end
function ∫ρz²dz(ρ::SphericalDensityProfile,z_eval::Float64,span::StepRangeLen)
    return 1. /z_eval * ∫ρrr²dr_scalar(ρ,z_eval,span)
end

function ∫ρrdr(ρ::SphericalDensityProfile,r_eval::Float64,span::StepRangeLen)
    dr = step(span)*one(eltype(ρ.density))
    r = r_eval.+span
    I = (ρ(ri) * ri for ri in r)
    return ∫(I,dr,length(r))
end

# for n3
function ∫ρrr²dr_scalar(ρ::SphericalDensityProfile,r_eval::Float64,span::StepRangeLen)
    dr = step(span)*one(eltype(ρ.density))
    r = r_eval.+span
    I = (ρ(ri) * ri * (span[end]^2 - (r_eval - ri)^2) for ri in r)
    return  ∫(I,dr,length(r))
end

# for n2_vec & n3_vec
function ∫ρrr²dr_vector(ρ::SphericalDensityProfile,r_eval::Float64,span::StepRangeLen)
    dr = step(span)*one(eltype(ρ.density))
    r = r_eval.+span
    I = (ρ(ri) * ri * (r_eval^2 - ri^2 + span[end]^2) for ri in r)
    return  ∫(I,dr,length(r))
end