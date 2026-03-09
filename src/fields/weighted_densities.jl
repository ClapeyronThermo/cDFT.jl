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

function SWeightedDensity(type::Symbol,width::Vector{Float64},ω, ngrid, backend::Backend)
    R = Adapt.adapt(backend, 2π.*width')
    # Reshape R based on the dimension of the system
    R = reshape(R, 1, length(width))
    nd = length(ngrid)

    if type != :∫ρzdz
        Ω = allocate(backend,ComplexF64,ngrid...,length(width))
    else
        Ω = allocate(backend,ComplexF64,ngrid...,length(ngrid),length(width))
    end

    if type == :∫ρdz
        for kk in CartesianIndices(ngrid)
            k = Tuple(kk)
            ω̄ = norm(@view(ω[k...,:]))
            Ω[k...,:] = 2*R .* (ω̄ .== 0.0) + 2*sin.(ω̄.*R)./ω̄ .*(ω̄ .!= 0.0)
        end
        Ω ./= 2π
    elseif type == :∫ρz²dz
        ω̄ = sqrt.(sum(abs2, ω, dims=nd+1)) 
        ω̄R   = ω̄ .* R                                 # (Nx,Ny,Nz,Nb)

        mask = ω̄ .== 0  
        Ω .= ifelse.(mask,
                4π*R.^3/3,                                    # ω̄=0 case
                4π./ω̄.^3 .*(sin.(ω̄R)-R.*ω̄.*cos.(ω̄R))        # ω̄≠0 case
            )
        Ω ./= (2π)^3
    elseif type == :ρ
        Ω .= 1.0 + 0im
    else
        error("Invalid type of field")
    end

    tmp = complex(Array(selectdim(Ω,nd+1,1)))
    plan = plan_fft!(tmp, 1:length(ngrid); num_threads=Threads.nthreads())
    iplan = inv(plan)
    return SWeightedDensity(type,width,Ω,plan,iplan)
end

function evaluate_field!(system::AbstractcDFTSystem,field::SWeightedDensity, ρ, n, in_buf, out_buf, P, iP)
    backend = system.options.device
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    if field.type == :ρ
        @. n = ρ*N_A
        return n
    end

    map = field.map
    for i in 1:nb
        ni = selectdim(n,nd+1,i)

        convolve!(ni, selectdim(ρ, nd+1, i), selectdim(map,nd+1,i), P, iP, in_buf)
    end
    synchronize(backend)
    @. n = real(n)*N_A
end

function integrate_field!(system::AbstractcDFTSystem, field::SWeightedDensity, profile, δfδρ_res, in_buf, P, iP)
    backend = system.options.device
    type = field.type
    ngrid = system.structure.ngrid
    nd = dimension(system)
    nb = size(profile,nd+1)


    if type == :ρ
        δfδρ_res .= profile
        return profile
    # else
    #     error("Invalid type of field")
    end

    map = field.map 

    for i in 1:nb
        convolve!(in_buf, selectdim(profile, nd+1, i), selectdim(map,nd+1,i), P, iP, in_buf)

        selectdim(δfδρ_res, nd+1, i) .+= in_buf
    end
    synchronize(backend)

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

function VWeightedDensity(type::Symbol,width::Vector{Float64},ω, ngrid, backend::Backend)
    R = Adapt.adapt(backend, 2π.*width')
    R = reshape(R, ntuple(i -> 1, length(ngrid))..., 1, length(width))
    nd = length(ngrid)


    if type != :∫ρzdz
        Ω = allocate(backend,ComplexF64,ngrid...,length(width))
    else
        Ω = allocate(backend,ComplexF64,ngrid...,length(ngrid),length(width))
    end

    if type == :∇ρ
        Ω = 1im .* ω
        Ω .*= (2π)
    elseif type == :∫ρzdz
        ω̄ = sqrt.(sum(abs2, ω, dims=nd+1)) 
        ω̄R   = ω̄ .* R                                 # (Nx,Ny,Nz,Nb)

        mask = ω̄ .== 0                                      # (Nx,Ny,Nz,1)  broadcasts over Nb

        Ω .= ifelse.(mask,
                4π*im*R,                                    # ω̄=0 case
                4π*im*ω./ω̄.^3 .*(sin.(ω̄R)-R.*ω̄.*cos.(ω̄R))        # ω̄≠0 case
            )
        # @time for kk in CartesianIndices(ngrid)
        #     k = Tuple(kk)
        #     ω̄ = norm(@view(ω[k...,:]))
        #     @. Ω[k...,:,:] = 0.0 - 4π*im*ω[k...,:]/ω̄^3*(sin(ω̄*R)-R*ω̄*cos(ω̄*R)) * (ω̄ != 0.0)
        # end
        # println("Time for ∫ρzdz: $t seconds")
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

    tmp = complex(Array(selectdim(selectdim(Ω,nd+1,1),nd+1,1)))
    plan = plan_fft!(tmp, 1:length(ngrid); num_threads=Threads.nthreads())
    iplan = inv(plan)
    return VWeightedDensity(type,width,Ω,plan,iplan)
end

function evaluate_field!(system::AbstractcDFTSystem,field::VWeightedDensity, ρ, nV, in_buf, out_buf, P, iP)
    backend = system.options.device

    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    map = field.map
    # @show eltype(nV)

    for i in 1:nb
        for j in 1:nd
            nVij = selectdim(selectdim(nV,nd+1,j),nd+1,i)
            mapij = selectdim(selectdim(map,nd+1,j),nd+1,i)

            convolve!(nVij, selectdim(ρ, nd+1, i), mapij, P, iP, in_buf)
        end
    end
    synchronize(backend)
    @. nV = real(nV)*N_A
end

function integrate_field!(system::AbstractcDFTSystem,field::VWeightedDensity, profile, δfδρ_res, in_buf, P, iP)
    backend = system.options.device
    type = field.type
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(profile,nd+2)

    map = field.map 

    for i in 1:nb
        for j in 1:nd
            convolve!(in_buf, selectdim(selectdim(profile,nd+1,j),nd+1,i), selectdim(selectdim(map,nd+1,j),nd+1,i), P, iP, in_buf)

            selectdim(δfδρ_res, nd+1, i) .-= in_buf
        end
    end
    synchronize(backend)
end