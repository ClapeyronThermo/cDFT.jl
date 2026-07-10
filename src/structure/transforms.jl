"""
    transform_eltype(structure, ::Type{FP})

Element type of the scratch buffer used to drive the radial/spatial transform for
`structure`: `Complex{FP}` for Cartesian (FFT-based), plain `FP` for spherical/
cylindrical (QDHT-based, real-valued throughout).
"""
transform_eltype(::DFTStructByCoord{Cartesian}, ::Type{FP}) where FP<:AbstractFloat = Complex{FP}
transform_eltype(::DFTStructByCoord{Cylindrical}, ::Type{FP}) where {FP<:AbstractFloat} = FP
transform_eltype(::DFTStructByCoord{Spherical}, ::Type{FP}) where {FP<:AbstractFloat} = FP

"""
    build_transform(structure, tmp, nd, backend)

Build the `(plan, iplan)` pair used by `convolve!` for `structure`: an FFTW plan/
inverse-plan pair for Cartesian structures, or the shared `Hankel.QDHT` object (used as
both "plan" and "iplan", see `convolve!`) for spherical/cylindrical structures.
"""
function build_transform(structure::DFTStructure, tmp::AbstractArray, nd::Int, backend::Backend)
    plan = backend isa CPU ? plan_fft!(tmp, 1:nd; num_threads=Threads.nthreads()) : plan_fft!(tmp, 1:nd)
    return plan, inv(plan)
end

function build_transform(structure::Union{DFTStructByCoord{Cylindrical},DFTStructByCoord{Spherical}} tmp::AbstractArray, nd::Int, backend::Backend)
    backend isa CPU || error("Spherical/cylindrical coordinate systems are CPU-only for now")
    Q = radial_transform(structure)
    return Q, Q
end

"""
    radial_derivative_matrix(r::AbstractVector)

Dense `N×N` finite-difference matrix approximating `d/dr` on the (possibly non-uniform)
grid `r`: 2nd-order-accurate central differences in the interior, 2nd-order one-sided
differences at the two boundary points. Used to obtain vector (gradient-type) weighted
densities from a scalar radial potential in spherical/cylindrical coordinates, where
`Hankel.QDHT` only provides a scalar (order-0) transform.
"""
function radial_derivative_matrix(r::AbstractVector{FP}) where FP<:AbstractFloat
    N = length(r)
    D = zeros(FP, N, N)
    for i in 2:N-1
        h1 = r[i] - r[i-1]
        h2 = r[i+1] - r[i]
        D[i,i-1] = -h2 / (h1*(h1+h2))
        D[i,i]   = (h2 - h1) / (h1*h2)
        D[i,i+1] = h1 / (h2*(h1+h2))
    end
    h1 = r[2] - r[1]
    h2 = r[3] - r[2]
    D[1,1] = -(2h1+h2) / (h1*(h1+h2))
    D[1,2] = (h1+h2) / (h1*h2)
    D[1,3] = -h1 / (h2*(h1+h2))
    h1 = r[N-1] - r[N-2]
    h2 = r[N] - r[N-1]
    D[N,N-2] = h2 / (h1*(h1+h2))
    D[N,N-1] = -(h1+h2) / (h1*h2)
    D[N,N]   = (2h2+h1) / (h2*(h1+h2))
    return D
end
