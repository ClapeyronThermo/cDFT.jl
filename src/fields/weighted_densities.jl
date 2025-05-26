"""
    SWeightedDensity(type::Symbol,width::Vector{Float64},map::Array{ComplexF64})

Generic `SWeightedDensity` type used to calculate the scalar weighted densities of the system. One must specify:
- `type`: The type of weighted density to be calculated. Options are `:∫ρdz` (``n_0``), `:∫ρzdz` (``n_v``), `:∫ρz²dz` (``n_3``), and `:ρ` (unweighted).
- `width`: The width of the weighted density profile.
- `map`: The Fourier transform of the weights.
- `plan`: The Fourier transform plan.
- `iplan`: The inverse Fourier transform plan.
"""
struct SWeightedDensity{M,P,iP} <: ScalarField 
    type::Symbol
    width::Vector{Float64}
    map::M
    plan::P
    iplan::iP
end

function SWeightedDensity(type::Symbol,width::Vector{Float64},ω::Array{Float64}, ngrid)
    
    R = 2π.*width'
    nd = length(ngrid)

    if type != :∫ρzdz
        Ω = zeros(ComplexF64,ngrid...,length(width))
    else
        Ω = zeros(ComplexF64,ngrid...,length(ngrid),length(width))
    end

    if type == :∫ρdz
        for kk in CartesianIndices(ngrid)
            k = Tuple(kk)
            ω̄ = norm(@view(ω[k...,:]))
            Ω[k...,:] = 2*R .* (ω̄ .== 0.0) + 2*sin.(ω̄.*R)./ω̄ .*(ω̄ .!= 0.0)
        end
        Ω ./= 2π
    elseif type == :∫ρz²dz
        for kk in CartesianIndices(ngrid)
            k = Tuple(kk)
            ω̄ = norm(@view(ω[k...,:]))
            Ω[k...,:] = 4π./ω̄.^3 .*(sin.(ω̄.*R)-R.*ω̄.*cos.(ω̄.*R)) .*(ω̄ .!= 0.0) + R.^3/3*4π .*(ω̄ .== 0.0)
        end
        # Ω = 4π./ω.^3 .*(sin.(ω.*R)-R.*ω.*cos.(ω.*R)) .*(ω .!= 0.0) + R.^3/3*4π .*(ω .== 0.0)
        Ω ./= (2π)^3
    elseif type == :ρ
        Ω = ones(ngrid...,length(width))
    else
        error("Invalid type of field")
    end

    plan = plan_fft(selectdim(Ω,nd+1,1), 1:length(ngrid))
    iplan = inv(plan)
    return SWeightedDensity(type,width,Ω,plan,iplan)
end

function evaluate_field(system::AbstractcDFTSystem,field::SWeightedDensity, ρ)
    structure = system.structure
    ngrid = structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    if field.type == :ρ
        return ρ.*N_A
    end

    map = field.map
    P = field.plan
    iP = field.iplan
    n = zeros(eltype(map),ngrid...,nb)

    for i in 1:nb
        ni = selectdim(n,nd+1,i)
        matmul!(ni,P,selectdim(ρ,nd+1,i))
        elmul!(ni,ni,selectdim(map,nd+1,i))
        matmul!(ni,iP,ni)
    end
    return real.(n).*N_A
end

function integrate_field(system::AbstractcDFTSystem,field::SWeightedDensity,profile)
    type = field.type
    ngrid = system.structure.ngrid
    nd = dimension(system)
    nb = size(profile,nd+1)


    if type == :ρ
        return profile
    # else
    #     error("Invalid type of field")
    end

    map = field.map 
    P = field.plan
    iP = field.iplan

    ∫field = zeros(eltype(map),ngrid...,nb)

    for i in 1:nb
        ∫fieldi = selectdim(∫field,nd+1,i)
        matmul!(∫fieldi,P,selectdim(profile,nd+1,i))
        elmul!(∫fieldi,∫fieldi,selectdim(map,nd+1,i))
        matmul!(∫fieldi,iP,∫fieldi)
    end
    return real.(∫field)
end

"""
    VWeightedDensity(type::Symbol,width::Vector{Float64},map::Array{ComplexF64})

Generic `VWeightedDensity` type used to calculate the vector weighted densities of the system. One must specify:
- `type`: The type of weighted density to be calculated. Options are `:∫ρdz` (``n_0``), `:∫ρzdz` (``n_v``), `:∫ρz²dz` (``n_3``), and `:ρ` (unweighted).
- `width`: The width of the weighted density profile.
- `map`: The Fourier transform of the weights.
- `plan`: The Fourier transform plan.
- `iplan`: The inverse Fourier transform plan.
"""
struct VWeightedDensity{M,P,iP} <: VectorField 
    type::Symbol
    width::Vector{Float64}
    map::M
    plan::P
    iplan::iP
end

function VWeightedDensity(type::Symbol,width::Vector{Float64},ω::Array{Float64}, ngrid)
    
    R = 2π.*width'
    nd = length(ngrid)

    # if type != :∫ρzdz
    #     Ω = zeros(ComplexF64,ngrid...,length(width))
    # else
    Ω = zeros(ComplexF64,ngrid...,length(ngrid),length(width))
    # end

    if type == :∇ρ
        for kk in CartesianIndices(ngrid)
            k = Tuple(kk)
            ω̄ = norm(@view(ω[k...,:]))
            Ω[k...,:] = ω[k...,:].*im
        end
        Ω .*= (2π)
    elseif type == :∫ρzdz
        for kk in CartesianIndices(ngrid)
            k = Tuple(kk)
            ω̄ = norm(@view(ω[k...,:]))
            Ω[k...,:,:] = @. 0.0 - 4π*im*ω[k...,:]/ω̄^3*(sin(ω̄*R)-R*ω̄*cos(ω̄*R)) *(ω̄ != 0.0)
        end
        # Ω = 4π*im./ω.^2 .*(sin.(ω.*R)-R.*ω.*cos.(ω.*R)) .*(ω .!= 0.0) .+ 0.0
        Ω ./= (2π)^3
    elseif type == :∫ρz²dz
        for kk in CartesianIndices(ngrid)
            k = Tuple(kk)
            ω̄ = norm(@view(ω[k...,:]))
            Ω[k...,:] = 4π./ω̄.^3 .*(sin.(ω̄.*R)-R.*ω̄.*cos.(ω̄.*R)) .*(ω̄ .!= 0.0) + R.^3/3*4π .*(ω̄ .== 0.0)
        end
        # Ω = 4π./ω.^3 .*(sin.(ω.*R)-R.*ω.*cos.(ω.*R)) .*(ω .!= 0.0) + R.^3/3*4π .*(ω .== 0.0)
        Ω ./= (2π)^3
    elseif type == :ρ
        Ω = ones(ngrid...,length(width))
    else
        error("Invalid type of field")
    end

    plan = plan_fft(selectdim(selectdim(Ω,nd+1,1),nd+1,1), 1:length(ngrid))
    iplan = inv(plan)
    return VWeightedDensity(type,width,Ω,plan,iplan)
end

function evaluate_field(system::AbstractcDFTSystem,field::VWeightedDensity, ρ)
    structure = system.structure
    ngrid = structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    if field.type == :ρ
        return ρ.*N_A
    end

    map = field.map
    P = field.plan
    iP = field.iplan
    nV = zeros(eltype(map),ngrid...,nd,nb)

    for i in 1:nb
        for j in 1:nd
            nVij = selectdim(selectdim(nV,nd+1,j),nd+1,i)
            matmul!(nVij,P,selectdim(ρ,nd+1,i))
            elmul!(nVij,nVij,selectdim(selectdim(map,nd+1,j),nd+1,i))
            matmul!(nVij,iP,nVij)
        end
    end
    return real.(nV).*N_A
end

function integrate_field(system::AbstractcDFTSystem,field::VWeightedDensity,profile)
    type = field.type
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(profile,nd+2)

    map = field.map 
    P = field.plan
    iP = field.iplan

    ∫field = zeros(eltype(map),ngrid...,nd,nb)
    # println(size(nV))

    for i in 1:nb
        for j in 1:nd
            ∫fieldij = selectdim(selectdim(∫field,nd+1,j),nd+1,i)
            matmul!(∫fieldij,P,selectdim(selectdim(profile,nd+1,j),nd+1,i))
            elmul!(∫fieldij,∫fieldij,selectdim(selectdim(map,nd+1,j),nd+1,i))
            matmul!(∫fieldij,iP,∫fieldij)
        end
        # ∫field[:,i] = prefactor*real.(ifft(fft(profile[:,i]).*map[:,i]))
    end
    return dropdims(sum(real.(∫field),dims=nd+1);dims=nd+1).*-1
end