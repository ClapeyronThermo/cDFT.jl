
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

function _∫(f::DensityProfile,dz::Number)
    ∑f = zero(typeof(dz))
    for i in 1:length(f.coeffs)
        ∑f += evalpoly(f.coords[i+1]-f.coords[i],(0.0,(f.coeffs[i]./(1,2,3,4))...))
    end
    return ∑f
end