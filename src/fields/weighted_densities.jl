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
    nd = length(ngrid)
    CT = eltype(ω)
    FP = real(CT)
    R = adapt_to_device(backend, FP, 2π.*width')
    # Reshape R based on the dimension of the system
    # R = reshape(R,length(width))
    R = reshape(R, ntuple(i -> 1, nd)..., length(width))

    if type != :∫ρzdz
        Ω = allocate(backend,CT,ngrid...,length(width))
    else
        Ω = allocate(backend,CT,ngrid...,length(ngrid),length(width))
    end

    if type == :∫ρdz
        ω̄ = sqrt.(sum(abs2, ω, dims=nd+1))

        ω̄R   = ω̄ .* R                                 # (Nx,Ny,Nz,Nb)

        mask = ω̄ .== 0

        Ω .= ifelse.(mask,
                2*R ,                                    # ω̄=0 case
                2*sin.(ω̄.*R)./ω̄        # ω̄≠0 case
            )
        Ω ./= FP(2π)
    elseif type == :∫ρz²dz
        ω̄ = sqrt.(sum(abs2, ω, dims=nd+1))
        ω̄R   = ω̄ .* R                                 # (Nx,Ny,Nz,Nb)

        mask = ω̄ .== 0
        Ω .= ifelse.(mask,
                FP(4π)*R.^3/3,                                    # ω̄=0 case
                FP(4π)./ω̄.^3 .*(sin.(ω̄R)-R.*ω̄.*cos.(ω̄R))        # ω̄≠0 case
            )
        Ω ./= FP(2π)^3
    elseif type == :ρ
        fill!(Ω, one(CT))
    else
        error("Invalid type of field")
    end

    tmp = complex(Array(selectdim(Ω,nd+1,1)))
    plan = plan_fft!(tmp, 1:length(ngrid); num_threads=Threads.nthreads())
    iplan = inv(plan)
    return SWeightedDensity(type,width,Ω,plan,iplan)
end

"""
    SWeightedDensity(type, width, ω::RadialFrequency, ngrid, backend)

Spherical/cylindrical (QDHT-based) counterpart of the Cartesian `SWeightedDensity`
constructor above. Reuses the same closed-form kernel formulas (they are exact 3D
isotropic Fourier transforms, valid regardless of the real-space coordinate system used
to sample them), substituting `ω.ω̄` for the Cartesian `ω̄ = sqrt.(sum(abs2,ω,dims=nd+1))`
and dropping the `ω̄=0` branch (QDHT never samples the origin in k-space). `map`/`plan`/
`iplan` are all real-valued (no `Complex` cast) since QDHT operates on real arrays.
"""
function SWeightedDensity(type::Symbol, width::Vector{Float64}, ω::RadialFrequency{FP}, ngrid, backend::Backend) where FP<:AbstractFloat
    N = ngrid[1]
    ω̄ = ω.ω̄
    R = FP.(2π .* width)

    Ω = allocate(backend, FP, N, length(width))
    if type == :∫ρdz
        for j in eachindex(width)
            @. Ω[:,j] = 2 * sin(ω̄*R[j]) / ω̄ / FP(2π)
        end
    elseif type == :∫ρz²dz
        for j in eachindex(width)
            @. Ω[:,j] = FP(4π) / ω̄^3 * (sin(ω̄*R[j]) - R[j]*ω̄*cos(ω̄*R[j])) / FP(2π)^3
        end
    elseif type == :ρ
        fill!(Ω, one(FP))
    else
        error("Invalid type of field")
    end

    Q = ω.Q
    return SWeightedDensity(type, width, Ω, Q, Q)
end

function evaluate_field!(system::AbstractcDFTSystem,field::SWeightedDensity, ρ, n, in_buf, out_buf, P, iP)
    backend = system.options.device
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)
    NA = eltype(ρ)(N_A)

    if field.type == :ρ
        @. n = ρ * NA
        return n
    end

    map = field.map
    for i in 1:nb
        ni = selectdim(n,nd+1,i)

        convolve!(ni, selectdim(ρ, nd+1, i), selectdim(map,nd+1,i), P, iP, in_buf)
    end
    # synchronize(backend)
    @. n = real(n) * NA
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
    # synchronize(backend)

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
    nd = length(ngrid)
    CT = eltype(ω)
    FP = real(CT)
    R = adapt_to_device(backend, FP, 2π.*width')
    R = reshape(R, ntuple(i -> 1, nd)..., 1, length(width))

    if type != :∫ρzdz
        Ω = allocate(backend,CT,ngrid...,length(width))
    else
        Ω = allocate(backend,CT,ngrid...,length(ngrid),length(width))
    end

    if type == :∇ρ
        Ω = 1im .* ω
        Ω .*= FP(2π)
    elseif type == :∫ρzdz
        ω̄ = sqrt.(sum(abs2, ω, dims=nd+1))
        ω̄R   = ω̄ .* R                                 # (Nx,Ny,Nz,Nb)

        mask = ω̄ .== 0                                      # (Nx,Ny,Nz,1)  broadcasts over Nb

        Ω .= ifelse.(mask,
                FP(4π)*im*R,                                    # ω̄=0 case
                FP(4π)*im*ω./ω̄.^3 .*(sin.(ω̄R)-R.*ω̄.*cos.(ω̄R))        # ω̄≠0 case
            )
        # @time for kk in CartesianIndices(ngrid)
        #     k = Tuple(kk)
        #     ω̄ = norm(@view(ω[k...,:]))
        #     @. Ω[k...,:,:] = 0.0 - 4π*im*ω[k...,:]/ω̄^3*(sin(ω̄*R)-R*ω̄*cos(ω̄*R)) * (ω̄ != 0.0)
        # end
        # println("Time for ∫ρzdz: $t seconds")
        # Ω = 4π*im./ω.^2 .*(sin.(ω.*R)-R.*ω.*cos.(ω.*R)) .*(ω .!= 0.0) .+ 0.0
        Ω ./= FP(2π)^3
    elseif type == :∫ρz²dz
        for kk in CartesianIndices(ngrid)
            k = Tuple(kk)
            ω̄ = FP(norm(@view(Array(ω)[k...,:])))
            Ω[k...,:] = FP(4π)./ω̄.^3 .*(sin.(ω̄.*R)-R.*ω̄.*cos.(ω̄.*R)) .*(ω̄ .!= 0) + R.^3/3*FP(4π) .*(ω̄ .== 0)
        end
        # Ω = 4π./ω.^3 .*(sin.(ω.*R)-R.*ω.*cos.(ω.*R)) .*(ω .!= 0.0) + R.^3/3*4π .*(ω .== 0.0)
        Ω ./= FP(2π)^3
    elseif type == :ρ
        fill!(Ω, one(CT))
    else
        error("Invalid type of field")
    end

    tmp = complex(Array(selectdim(selectdim(Ω,nd+1,1),nd+1,1)))
    plan = plan_fft!(tmp, 1:length(ngrid); num_threads=Threads.nthreads())
    iplan = inv(plan)
    return VWeightedDensity(type,width,Ω,plan,iplan)
end

"""
    VWeightedDensity(type, width, ω::RadialFrequency, ngrid, backend)

Spherical/cylindrical (QDHT-based) counterpart of the Cartesian `VWeightedDensity`
constructor above, for the `:∫ρzdz` vector weighted density (used by every FMT-based
model's hard-sphere functional).

The Cartesian kernel has the form `Ω_j(k) = i·k̂_j·H(ω̄)`, i.e. it is exactly the Fourier
multiplier for `∇` of the radially-symmetric scalar potential `M(r)` whose Fourier
transform is `H(ω̄)` (the same closed form used by `SWeightedDensity(:∫ρz²dz,...)`).
Since `Hankel.QDHT` only provides a *scalar* (order-0) transform, this is implemented as:
scalar QDHT-convolve `ρ` with `H` to get `M(r)`, then differentiate `M` in real space
(via `radial_derivative_matrix`) to get the vector's radial component (`∇M(r)=M'(r)r̂`
exactly, by radial symmetry).

Rather than deriving a separate closed-form adjoint for the reverse pass, the full
forward linear operator `ρ ↦ n_v` (QDHT, multiply by `H`, inverse QDHT, differentiate) is
materialized as a dense `N×N` matrix per bead (cheap at `N≲200`, done once here at
construction time), and its *exact* matrix transpose is used for the adjoint in
`integrate_field!` — this guarantees the forward/adjoint pair are consistent by
construction, rather than relying on a hand-derived (and easy to get subtly wrong)
weighted-inner-product adjoint formula.
"""
function VWeightedDensity(type::Symbol, width::Vector{Float64}, ω::RadialFrequency{FP}, ngrid, backend::Backend) where FP<:AbstractFloat
    type == :∫ρzdz || error("Only :∫ρzdz vector weighted densities are supported for spherical/cylindrical coordinates")
    N = ngrid[1]
    Q = ω.Q
    ω̄ = ω.ω̄
    Rk = FP.(2π .* width)
    nb = length(width)

    D = radial_derivative_matrix(FP.(Q.r))

    T = Array{FP}(undef, N, N, nb)
    e    = zeros(FP, N)
    tmp1 = similar(e)
    tmp2 = similar(e)
    for j in 1:nb
        Hj = @. FP(4π) / ω̄^3 * (sin(ω̄*Rk[j]) - Rk[j]*ω̄*cos(ω̄*Rk[j])) / FP(2π)^3
        for k in 1:N
            fill!(e, 0); e[k] = 1
            LinearAlgebra.mul!(tmp1, Q, e)
            tmp1 .*= Hj
            LinearAlgebra.ldiv!(tmp2, Q, tmp1)
            T[:,k,j] = D * tmp2
        end
    end

    return VWeightedDensity(type, width, T, Q, Q)
end

function evaluate_field!(system::AbstractcDFTSystem, field::VWeightedDensity, ρ, nV, in_buf, out_buf, P::Hankel.QDHT, iP::Hankel.QDHT)
    field.type == :∫ρzdz || error("Unsupported vector weighted density type for spherical/cylindrical coordinates: $(field.type)")
    nd = length(system.structure.ngrid)
    nb = size(ρ,nd+1)
    NA = eltype(ρ)(N_A)
    T = field.map

    for i in 1:nb
        nVi = selectdim(selectdim(nV,nd+1,1),nd+1,i)
        LinearAlgebra.mul!(nVi, view(T,:,:,i), selectdim(ρ, nd+1, i))
    end
    @. nV = real(nV) * NA
end

function integrate_field!(system::AbstractcDFTSystem, field::VWeightedDensity, profile, δfδρ_res, in_buf, P::Hankel.QDHT, iP::Hankel.QDHT)
    field.type == :∫ρzdz || error("Unsupported vector weighted density type for spherical/cylindrical coordinates: $(field.type)")
    nd = length(system.structure.ngrid)
    nb = size(profile,nd+2)
    T = field.map

    for i in 1:nb
        δnVi = selectdim(selectdim(profile,nd+1,1),nd+1,i)
        selectdim(δfδρ_res, nd+1, i) .+= transpose(view(T,:,:,i)) * δnVi
    end
end

function evaluate_field!(system::AbstractcDFTSystem,field::VWeightedDensity, ρ, nV, in_buf, out_buf, P, iP)
    backend = system.options.device

    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)
    NA = eltype(ρ)(N_A)

    map = field.map
    # @show eltype(nV)

    for i in 1:nb
        for j in 1:nd
            nVij = selectdim(selectdim(nV,nd+1,j),nd+1,i)
            mapij = selectdim(selectdim(map,nd+1,j),nd+1,i)

            convolve!(nVij, selectdim(ρ, nd+1, i), mapij, P, iP, in_buf)
        end
    end
    # synchronize(backend)
    @. nV = real(nV) * NA
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
    # synchronize(backend)
end