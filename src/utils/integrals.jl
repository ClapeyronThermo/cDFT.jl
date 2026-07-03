
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
        ∑f += prod(coef)*f[k...]
    end

    ∑f *= prod(dz./3)
    return ∑f
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