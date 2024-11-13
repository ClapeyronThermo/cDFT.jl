"""
    SWeightedDensity(type::Symbol,width::Vector{Float64},map::Array{ComplexF64})

Generic `SWeightedDensity` type used to calculate the scalar weighted densities of the system. One must specify:
- `type`: The type of weighted density to be calculated. Options are `:∫ρdz` (``n_0``), `:∫ρzdz` (``n_v``), `:∫ρz²dz` (``n_3``), and `:ρ` (unweighted).
- `width`: The width of the weighted density profile.
- `map`: The Fourier transform of the weights.
- `plan`: The Fourier transform plan.
- `iplan`: The inverse Fourier transform plan.
"""
struct SWeightedDensity <: ScalarField 
    type::Symbol
    width::Vector{Float64}
    map::Array{ComplexF64}
    plan::Plan
    iplan::Plan
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
        for k in Iterators.product([1:ngrid[i] for i in 1:length(ngrid)]...)
            ω̄ = norm(ω[k...,:])
            Ω[k...,:] = 2*R .* (ω̄ .== 0.0) + 2*sin.(ω̄.*R)./ω̄ .*(ω̄ .!= 0.0)
        end
        Ω ./= 2π
    elseif type == :∫ρz²dz
        for k in Iterators.product([1:ngrid[i] for i in 1:length(ngrid)]...)
            ω̄ = norm(ω[k...,:])
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

function evaluate_field(system::DFTSystem,field::SWeightedDensity, ρ)
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
        matmul!(selectdim(n,nd+1,i),P,selectdim(ρ,nd+1,i))
        elmul!(selectdim(n,nd+1,i),selectdim(n,nd+1,i),selectdim(map,nd+1,i))
        matmul!(selectdim(n,nd+1,i),iP,selectdim(n,nd+1,i))
    end
    return real.(n).*N_A
end

function integrate_field(system::DFTSystem,field::SWeightedDensity,profile)
    type = field.type
    ngrid = system.structure.ngrid
    nd = length(ngrid)
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
        matmul!(selectdim(∫field,nd+1,i),P,selectdim(profile,nd+1,i))
        elmul!(selectdim(∫field,nd+1,i),selectdim(∫field,nd+1,i),selectdim(map,nd+1,i))
        matmul!(selectdim(∫field,nd+1,i),iP,selectdim(∫field,nd+1,i))
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
struct VWeightedDensity <: VectorField 
    type::Symbol
    width::Vector{Float64}
    map::Array{ComplexF64}
    plan::Plan
    iplan::Plan
end

function VWeightedDensity(type::Symbol,width::Vector{Float64},ω::Array{Float64}, ngrid)
    
    R = 2π.*width
    nd = length(ngrid)

    if type != :∫ρzdz
        Ω = zeros(ComplexF64,ngrid...,length(width))
    else
        Ω = zeros(ComplexF64,ngrid...,length(width),length(ngrid))
    end

    if type == :∫ρdz
        for k in Iterators.product([1:ngrid[i] for i in 1:length(ngrid)]...)
            ω̄ = norm(ω[k...,:])
            Ω[k...,:] = 2*R .* (ω̄ .== 0.0) + 2*sin.(ω̄.*R)./ω̄ .*(ω̄ .!= 0.0)
        end
        Ω ./= 2π
    elseif type == :∫ρzdz
        for k in Iterators.product([1:ngrid[i] for i in 1:length(ngrid)]...)
            ω̄ = norm(ω[k...,:])
            Ω[k...,:,:] = @. 0.0 - 4π*im*abs(ω[k...,:]')/ω̄^3*(sin(ω̄*R)-R*ω̄*cos(ω̄*R)) *(ω̄ != 0.0)
        end
        # Ω = 4π*im./ω.^2 .*(sin.(ω.*R)-R.*ω.*cos.(ω.*R)) .*(ω .!= 0.0) .+ 0.0
        Ω ./= (2π)^3
    elseif type == :∫ρz²dz
        for k in Iterators.product([1:ngrid[i] for i in 1:length(ngrid)]...)
            ω̄ = norm(ω[k...,:])
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

function evaluate_field(system::DFTSystem,field::VWeightedDensity, ρ)
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
    nV = zeros(eltype(map),ngrid...,nb,nd)

    for i in 1:nb
        for j in 1:nd
            matmul!(selectdim(selectdim(nV,nd+1,i),nd+1,j),P,selectdim(ρ,nd+1,i))
            elmul!(selectdim(selectdim(nV,nd+1,i),nd+1,j),selectdim(selectdim(nV,nd+1,i),nd+1,j),selectdim(selectdim(map,nd+1,i),nd+1,j))
            matmul!(selectdim(selectdim(nV,nd+1,i),nd+1,j),iP,selectdim(selectdim(nV,nd+1,i),nd+1,j))
        end
    end
    return real.(nV).*N_A
end

function integrate_field(system::DFTSystem,field::VWeightedDensity,profile,nV)
    type = field.type
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(profile,nd+1)

    if type == :∫ρdz || type == :∫ρz²dz
        prefactor = 1
    elseif type == :∫ρzdz
        prefactor = -1
    elseif type == :ρ
        return profile
    else
        error("Invalid type of field")
    end

    map = field.map 
    P = field.plan
    iP = field.iplan

    ∫field = zeros(eltype(map),ngrid...,nb,nd)
    # println(size(nV))
    # println(size(profile))

    for i in 1:nb
        for j in 1:nd
            matmul!(selectdim(selectdim(∫field,nd+1,i),nd+1,j),P,selectdim(profile,nd+1,i).*abs.(selectdim(selectdim(nV,nd+1,i),nd+1,j)))
            elmul!(selectdim(selectdim(∫field,nd+1,i),nd+1,j),selectdim(selectdim(∫field,nd+1,i),nd+1,j),selectdim(selectdim(map,nd+1,i),nd+1,j))
            matmul!(selectdim(selectdim(∫field,nd+1,i),nd+1,j),iP,selectdim(selectdim(∫field,nd+1,i),nd+1,j))
        end
        # ∫field[:,i] = prefactor*real.(ifft(fft(profile[:,i]).*map[:,i]))
    end
    return dropdims(sum(real.(∫field),dims=nd+2);dims=nd+2).*-2
end