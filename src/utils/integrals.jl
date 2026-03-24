
"""
    ∫(f,dz)

Integrates a collection of points `f`, with constant `dz`, using simpson rule.

"""
∫(f,dz) = _∫(f,dz)
#function _∫(f::AbstractArray,dz::Number,lastidx)
#    return 1/3*dz*(f[1]+f[end]+4*sum(@view(f[2:2:end-1]))+2*sum(@view(f[3:2:end-1])))
#end

function _∫(f::Array{Float64},dz)
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

function convolve!(result, profile, kernel, P, iP, buf_r, buf_c)
    buf_r .= profile                   # copy real profile into real input buffer
    mul!(buf_c, P,  buf_r)             # R2C FFT: buf_r → buf_c (no allocation)
    buf_c .*= kernel                   # multiply by half-complex kernel in-place
    mul!(buf_r, iP, buf_c)             # C2R IFFT: buf_c → buf_r (no allocation)
    result .= buf_r                    # copy real result into output (no temporary)
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
function trapz_weights(ngrid::Tuple, dz, device)
    w = fill(prod(dz), ngrid...)
    return Adapt.adapt(device, w)
end

"""
    simpson_weights(ngrid, dz, device)

Precompute the multi-dimensional composite Simpson quadrature weight array on `device`.
Returns an array of shape `ngrid` such that `sum(f .* weights)` equals `∫(Array(f), dz)`
for any array `f` of the same shape, without requiring `f` to be on the CPU.

The weights replicate the coefficient pattern of `_∫` (1/4/2 pattern per dimension)
as an outer product, multiplied by `prod(dz ./ 3)`. Requires odd N in each dimension.
"""
function simpson_weights(ngrid::Tuple, dz, device)
    nd = length(ngrid)
    # Build 1D weight vector for each dimension
    w1d = map(1:nd) do d
        n = ngrid[d]
        h = dz[d]
        w = Vector{Float64}(undef, n)
        w[1]   = 1.0
        w[end] = 1.0
        for i in 2:n-1
            w[i] = i % 2 == 0 ? 2.0 : 4.0
        end
        w .*= h / 3
        w
    end
    # Outer product: broadcast each 1D vector along its own dimension
    weights = ones(Float64, ngrid...)
    for d in 1:nd
        shape = ntuple(i -> i == d ? ngrid[d] : 1, nd)
        weights .*= reshape(w1d[d], shape...)
    end
    return Adapt.adapt(device, weights)
end