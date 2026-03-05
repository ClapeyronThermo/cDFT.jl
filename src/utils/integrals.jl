
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

function convolve!(result, profile, kernel, P, iP, buf)
    copyto!(buf, complex.(profile))
    P * buf                            # in-place: buf is overwritten with FFT result
    elmul!(buf,buf,kernel)
    iP * buf              # inverse transform on contiguous buffer
    copyto!(result, real.(buf))      # copy result back
end