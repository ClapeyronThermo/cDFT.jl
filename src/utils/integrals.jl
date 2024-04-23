
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

function ∫ρdz(structure::DFTStructure1DCart,ρ::DensityProfile,z_eval::Float64,span::Float64)
    I = 0.
    
    z1 = z_eval-span
    z2 = z_eval+span

    idx1 = _binary_search_interval(ρ, z1)
    idx2 = _binary_search_interval(ρ, z2)

    if idx1 == 0
        I += ρ.boundary_conditions[1].value*(ρ.coords[1]-z1)
    else
        I += evalpoly(ρ.coords[idx1+1]-ρ.coords[idx1],(0.0,(ρ.coeffs[idx1]./(1,2,3,4))...))
        I -= evalpoly(z1-ρ.coords[idx1],(0.0,(ρ.coeffs[idx1]./(1,2,3,4))...))
    end

    if idx2 == structure.ngrid
        I += ρ.boundary_conditions[2].value*(z2-ρ.coords[end])
    else
        I += evalpoly(z2-ρ.coords[idx2],(0.0,(ρ.coeffs[idx2]./(1,2,3,4))...))
    end

    for i in idx1+1:idx2-1
        I += evalpoly(ρ.coords[i+1]-ρ.coords[i],(0.0,(ρ.coeffs[i]./(1,2,3,4))...))
    end

    return I
end

function ∫ρdz(structure::DFTStructure1DSphr,ρ::DensityProfile,r_eval::Float64,span::Float64)
    I = 0.
    
    r1 = r_eval-span
    r2 = r_eval+span

    idx1 = _binary_search_interval(ρ, r1)
    idx2 = _binary_search_interval(ρ, r2)

    if idx1 == 0
        I += ρ.boundary_conditions[1].value*((ρ.coords[1]^2-r1^2)/2)
    else
        coeff1 = ((ρ.coeffs[idx1].*ρ.coords[idx1])...,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx1])...)
        coeff = coeff2.+coeff1

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5)

        I += evalpoly(ρ.coords[idx1+1]-ρ.coords[idx1],coeff)
        I -= evalpoly(r1-ρ.coords[idx1],coeff)
    end

    if idx2 == structure.ngrid
        I += ρ.boundary_conditions[2].value*((r2^2-ρ.coords[end]^2)/2)
    else
        coeff1 = ((ρ.coeffs[idx2].*ρ.coords[idx2])...,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx2])...)
        coeff = coeff2.+coeff1

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5)

        I += evalpoly(r2-ρ.coords[idx2],coeff)
    end

    for i in idx1+1:idx2-1
        coeff1 = ((ρ.coeffs[i].*ρ.coords[i])...,0.0)
        coeff2 = (0.0,(ρ.coeffs[i])...)
        coeff = coeff2.+coeff1

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5)
        I += evalpoly(ρ.coords[i+1]-ρ.coords[i],coeff)
    end

    return I/r_eval
end

function ∫ρzdz(structure::DFTStructure1DCart,ρ::DensityProfile,z_eval::Float64,span::Float64)
    I = 0.
    
    z1 = z_eval-span
    z2 = z_eval+span

    idx1 = _binary_search_interval(ρ, z1)
    idx2 = _binary_search_interval(ρ, z2)

    if idx1 == 0
        I += ((ρ.coords[1]^2 - z1^2)/2 - z_eval*(ρ.coords[1]-z1))*ρ.boundary_conditions[1].value
    else
        coeff1 = ((ρ.coeffs[idx1].*(z_eval-ρ.coords[idx1]))...,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx1])...)
        coeff = coeff2.-coeff1

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5)

        I += evalpoly(ρ.coords[idx1+1]-ρ.coords[idx1],coeff)
        I -= evalpoly(z1-ρ.coords[idx1],coeff)
    end

    if idx2 == structure.ngrid
        I += ((z2^2-ρ.coords[end]^2)/2 - z_eval*(z2-ρ.coords[end]))*ρ.boundary_conditions[2].value
    else
        coeff1 = ((ρ.coeffs[idx2].*(z_eval-ρ.coords[idx2]))...,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx2])...)
        coeff = coeff2.-coeff1
        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5)

        I += evalpoly(z2-ρ.coords[idx2],coeff)
    end

    for i in idx1+1:idx2-1
        coeff1 = ((ρ.coeffs[i].*(z_eval-ρ.coords[i]))...,0.0)
        coeff2 = (0.0,(ρ.coeffs[i])...)
        coeff = coeff2.-coeff1
        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5)

        I += evalpoly(ρ.coords[i+1]-ρ.coords[i],coeff)
    end

    return I
end

function ∫ρzdz(structure::DFTStructure1DSphr,ρ::DensityProfile,r_eval::Float64,span::Float64)
    I = 0.
    r1 = r_eval-span
    r2 = r_eval+span

    idx1 = _binary_search_interval(ρ, r1)
    idx2 = _binary_search_interval(ρ, r2)


    if idx1 == 0
        I += ((span^2+r_eval^2)*(ρ.coords[1]^2-r1^2)/2-(ρ.coords[1]^4-r1^4)/4)*ρ.boundary_conditions[1].value
    else
        coeff1 = ((ρ.coeffs[idx1].*ρ.coords[idx1].*(span^2+r_eval^2-ρ.coords[idx1]^2))...,0.0,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx1].*(span^2+r_eval^2-3*ρ.coords[idx1]^2))...,0.0,0.0)
        coeff3 = (0.0,0.0,(3 .*ρ.coeffs[idx1].*ρ.coords[idx1])...,0.0)
        coeff4 = (0.0,0.0,0.0,(ρ.coeffs[idx1])...)
        coeff = coeff1.+coeff2.-coeff3.-coeff4

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6,7)

        I += evalpoly(ρ.coords[idx1+1]-ρ.coords[idx1],coeff)
        I -= evalpoly(r1-ρ.coords[idx1],coeff)
    end

    if idx2 == structure.ngrid
        I += ((span^2+r_eval^2)*(r2^2-ρ.coords[end]^2)/2-(r2^4-ρ.coords[end]^4)/4)*ρ.boundary_conditions[2].value
    else
        coeff1 = ((ρ.coeffs[idx2].*ρ.coords[idx2].*(span^2+r_eval^2-ρ.coords[idx2]^2))...,0.0,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx2].*(span^2+r_eval^2-3*ρ.coords[idx2]^2))...,0.0,0.0)
        coeff3 = (0.0,0.0,(3 .*ρ.coeffs[idx2].*ρ.coords[idx2])...,0.0)
        coeff4 = (0.0,0.0,0.0,(ρ.coeffs[idx2])...)
        coeff = coeff1.+coeff2.-coeff3.-coeff4

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6,7)

        I += evalpoly(r2-ρ.coords[idx2],coeff)
    end

    for i in idx1+1:idx2-1
        coeff1 = ((ρ.coeffs[i].*ρ.coords[i].*(span^2+r_eval^2-ρ.coords[i]^2))...,0.0,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[i].*(span^2+r_eval^2-3*ρ.coords[i]^2))...,0.0,0.0)
        coeff3 = (0.0,0.0,(3 .*ρ.coeffs[i].*ρ.coords[i])...,0.0)
        coeff4 = (0.0,0.0,0.0,(ρ.coeffs[i])...)
        coeff = coeff1.+coeff2.-coeff3.-coeff4

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6,7)

        I += evalpoly(ρ.coords[i+1]-ρ.coords[i],coeff)
    end
    # println(I/r_eval)
    return -I/r_eval^2/2
end

function ∫ρz²dz(structure::DFTStructure1DCart,ρ::DensityProfile,z_eval::Float64,span::Float64)
    I = 0.
    
    z1 = z_eval-span
    z2 = z_eval+span

    idx1 = _binary_search_interval(ρ, z1)
    idx2 = _binary_search_interval(ρ, z2)

    if idx1 == 0
        I += ((span^2-z_eval^2)*((ρ.coords[1]-z1))
           + 2*z_eval*(ρ.coords[1]^2 - z1^2)/2
           - (ρ.coords[1]^3 - z1^3)/3)*ρ.boundary_conditions[1].value
    else
        coeff1 = ((ρ.coeffs[idx1].*(span^2-(z_eval-ρ.coords[idx1])^2))...,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx1].*2 .*(z_eval-ρ.coords[idx1]))...,0.0)
        coeff3 = (0.0,0.0,(ρ.coeffs[idx1])...)
        coeff = coeff1.+coeff2.-coeff3

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6)

        I += evalpoly(ρ.coords[idx1+1]-ρ.coords[idx1],coeff)
        I -= evalpoly(z1-ρ.coords[idx1],coeff)
    end

    if idx2 == structure.ngrid
        I += ((span^2-z_eval^2)*((z2-ρ.coords[end]))
           + 2*z_eval*(z2^2-ρ.coords[end]^2)/2
           - (z2^3-ρ.coords[end]^3)/3)*ρ.boundary_conditions[2].value
    else
        coeff1 = ((ρ.coeffs[idx2].*(span^2-(z_eval-ρ.coords[idx2])^2))...,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx2].*2 .*(z_eval-ρ.coords[idx2]))...,0.0)
        coeff3 = (0.0,0.0,(ρ.coeffs[idx2])...)
        coeff = coeff1.+coeff2.-coeff3

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6)

        I += evalpoly(z2-ρ.coords[idx2],coeff)
    end

    for i in idx1+1:idx2-1
        coeff1 = ((ρ.coeffs[i].*(span^2-(z_eval-ρ.coords[i])^2))...,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[i].*2 .*(z_eval-ρ.coords[i]))...,0.0)
        coeff3 = (0.0,0.0,(ρ.coeffs[i])...)
        coeff = coeff1.+coeff2.-coeff3

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6)

        I += evalpoly(ρ.coords[i+1]-ρ.coords[i],coeff)
    end

    return I*π
end

function ∫ρz²dz(structure::DFTStructure1DSphr,ρ::DensityProfile,r_eval::Float64,span::Float64)
    I = 0.
    
    r1 = r_eval-span
    r2 = r_eval+span

    idx1 = _binary_search_interval(ρ, r1)
    idx2 = _binary_search_interval(ρ, r2)

    if idx1 == 0
        I += ((span^2-r_eval^2)*(ρ.coords[1]^2-r1^2)/2
           + 2*r_eval*(ρ.coords[1]^3 - r1^3)/3
           - (ρ.coords[1]^4 - r1^4)/4)*ρ.boundary_conditions[1].value
    else
        coeff1 = ((ρ.coeffs[idx1].*(span^2-(r_eval-ρ.coords[idx1])^2).*ρ.coords[idx1])...,0.0,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx1].*((span^2-r_eval^2)+4*ρ.coords[idx1]*r_eval-3*ρ.coords[idx1]^2))...,0.0,0.0)
        coeff3 = (0.0,0.0,(ρ.coeffs[idx1].*(2*r_eval-3*ρ.coords[idx1]))...,0.0)
        coeff4 = (0.0,0.0,0.0,(ρ.coeffs[idx1])...)
        coeff = coeff1.+coeff2.+coeff3.-coeff4

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6,7)

        I += evalpoly(ρ.coords[idx1+1]-ρ.coords[idx1],coeff)
        I -= evalpoly(r1-ρ.coords[idx1],coeff)
    end

    if idx2 == structure.ngrid
        I += ((span^2-r_eval^2)*(r2^2-ρ.coords[end]^2)/2
           + 2*r_eval*(r2^3-ρ.coords[end]^3)/3
           - (r2^4-ρ.coords[end]^4)/4)*ρ.boundary_conditions[2].value
    else
        coeff1 = ((ρ.coeffs[idx2].*(span^2-(r_eval-ρ.coords[idx2])^2).*ρ.coords[idx2])...,0.0,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[idx2].*((span^2-r_eval^2)+4*ρ.coords[idx2]*r_eval-3*ρ.coords[idx2]^2))...,0.0,0.0)
        coeff3 = (0.0,0.0,(ρ.coeffs[idx2].*(2*r_eval-3*ρ.coords[idx2]))...,0.0)
        coeff4 = (0.0,0.0,0.0,(ρ.coeffs[idx2])...)
        coeff = coeff1.+coeff2.+coeff3.-coeff4

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6,7)

        I += evalpoly(r2-ρ.coords[idx2],coeff)
    end

    for i in idx1+1:idx2-1
        coeff1 = ((ρ.coeffs[i].*(span^2-(r_eval-ρ.coords[i])^2).*ρ.coords[i])...,0.0,0.0,0.0)
        coeff2 = (0.0,(ρ.coeffs[i].*((span^2-r_eval^2)+4*ρ.coords[i]*r_eval-3*ρ.coords[i]^2))...,0.0,0.0)
        coeff3 = (0.0,0.0,(ρ.coeffs[i].*(2*r_eval-3*ρ.coords[i]))...,0.0)
        coeff4 = (0.0,0.0,0.0,(ρ.coeffs[i])...)
        coeff = coeff1.+coeff2.+coeff3.-coeff4

        coeff = (0.0,(coeff)...)
        coeff = coeff./(1,1,2,3,4,5,6,7)

        I += evalpoly(ρ.coords[i+1]-ρ.coords[i],coeff)
    end

    return I*π/r_eval
end