
"""
    ∫(f,dz)

Integrates a collection of points `f`, with constant `dz`, using simpson rule.

"""
∫(f,dz) = _∫(f,dz)

"""
    ∫(f, structure::DFTStructure)

Coordinate-aware integral of `f` (sampled on `structure`'s grid) over the physical
domain. For Cartesian structures this is Simpson's rule with the structure's uniform
grid spacing (unchanged behavior, just re-routed through `structure_dz`). For
spherical/cylindrical structures, this uses `Hankel.integrateR` (the radial quadrature
matched to the structure's `QDHT`, which already incorporates the `r`/`r²` Jacobian)
times the angular factor `Hankel.integrateR` itself omits (`4π` for a full sphere, `2π`
for a cylinder's circular cross-section).
"""
∫(f, structure::DFTStructureCart) = ∫(f, structure_dz(structure))

function ∫(f, structure::Union{DFTStructureSphr,DFTStructureCyl})
    Q = radial_transform(structure)
    angular = structure isa DFTStructureSphr ? 4π : 2π
    return angular * Hankel.integrateR(f, Q)
end
#function _∫(f::AbstractArray,dz::Number,lastidx)
#    return 1/3*dz*(f[1]+f[end]+4*sum(@view(f[2:2:end-1]))+2*sum(@view(f[3:2:end-1])))
#end

_∫(f::AbstractArray, dz) = _∫(Array(f), dz)

function _∫(f::Array{T},dz) where T<:Real
    ∑f = zero(typeof(first(dz)))
    for i in CartesianIndices(size(f))
        k = Tuple(i)
        # check if the indices in each dimension is even or odd
        coef = (k.==1 .|| k.==size(f)) .+ (2 .*(k .% 2 .== 0) .+ 4 .*(k .% 2 .!= 0)).*.!(k.==1 .|| k.==size(f))
        ∑f += FT(prod(coef)) * FT(f[k...])
    end

    ∑f *= FT(prod(dz ./ 3))
    return ∑f
end

"""
    trapz_weights(ngrid, dz, device)

Precompute the multi-dimensional periodic trapezoidal quadrature weight array on `device`.
Returns an array of shape `ngrid` where every element equals `prod(dz)`, so that
`sum(f .* weights)` approximates `∫ f dV` using the periodic composite trapezoidal rule.

For smooth periodic functions (e.g. SCFT density profiles in a periodic box) this rule
converges spectrally (exponentially in N) via the Euler-Maclaurin theorem, compared to
O(h⁴) for Simpson's rule. Works for any N (odd or even).
"""
function trapz_weights(ngrid::Tuple, dz, options::DFTOptions)
    FT = float_type(options)
    w = fill(FT(prod(dz)), ngrid...)
    return Adapt.adapt(options.device, w)
end

"""
    simpson_weights(ngrid, dz, device)

Precompute the multi-dimensional composite Simpson quadrature weight array on `device`.
Returns an array of shape `ngrid` such that `sum(f .* weights)` equals `∫(Array(f), dz)`
for any array `f` of the same shape, without requiring `f` to be on the CPU.

The weights replicate the coefficient pattern of `_∫` (1/4/2 pattern per dimension)
as an outer product, multiplied by `prod(dz ./ 3)`. Requires odd N in each dimension.
"""
function simpson_weights(ngrid::Tuple, dz, options::DFTOptions)
    nd = length(ngrid)
    FT = float_type(options)
    # Build 1D weight vector for each dimension
    w1d = map(1:nd) do d
        n = ngrid[d]
        h = dz[d]
        w = Vector{FT}(undef, n)
        w[1]   = FT(1)
        w[end] = FT(1)
        for i in 2:n-1
            w[i] = i % 2 == 0 ? FT(2) : FT(4)
        end
        w .*= FT(h / 3)
        w
    end
    # Outer product: broadcast each 1D vector along its own dimension
    weights = ones(FT, ngrid...)
    for d in 1:nd
        shape = ntuple(i -> i == d ? ngrid[d] : 1, nd)
        weights .*= reshape(w1d[d], shape...)
    end
    return Adapt.adapt(options.device, weights)
end
function convolve!(result, profile, kernel, P, iP, buf)
    if profile !== buf
        buf .= complex.(profile)
    end
    P * buf
    elmul!(buf, buf, kernel)
    iP * buf
    result .= real.(buf)
end

"""
    convolve!(result, profile, kernel, P::Hankel.QDHT, iP::Hankel.QDHT, buf)

Radial (spherical/cylindrical) analogue of the FFT-based `convolve!` above, using a
quasi-discrete Hankel transform in place of the FFT. Unlike `plan_fft!`, `Hankel.QDHT`'s
`mul!`/`ldiv!` cannot alias their input/output arrays, so a distinct scratch array `tmp`
is used for the k-space intermediate. `P` and `iP` are always the same `QDHT` object
(see `build_transform`) — there is no separate "inverse plan" the way `inv(plan_fft!(...))`
is needed for FFTW.
"""
function convolve!(result, profile, kernel, P::Hankel.QDHT, iP::Hankel.QDHT, buf)
    if profile !== buf
        buf .= profile
    end
    tmp = similar(buf)
    LinearAlgebra.mul!(tmp, P, buf)
    tmp .*= kernel
    LinearAlgebra.ldiv!(buf, iP, tmp)
    result .= buf
end
