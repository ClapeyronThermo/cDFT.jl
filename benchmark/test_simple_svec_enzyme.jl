using CUDA, Enzyme, KernelAbstractions, StaticArrays

@inline function simple_f_v(v)
    return v[1] * v[2]
end

@kernel function simple_kernel!(dx, @Const(x), @Const(y))
    i = @index(Global)
    v = SVector{2, Float64}(x[i], y[i])
    # Test if exactly ONE Active argument works, even if it is an SVector
    adj = Enzyme.autodiff_deferred(Reverse, Const(simple_f_v), Active, Active(v))
    # adj[1] should be the SVector gradient
    dx[i] = adj[1][1]
end

function test_simple()
    if !CUDA.functional(); return; end
    backend = CUDABackend()
    x = CUDA.fill(3.0, 10)
    y = CUDA.fill(2.0, 10)
    dx = CUDA.fill(0.0, 10)
    kernel = simple_kernel!(backend)
    kernel(dx, x, y, ndrange=10)
    KernelAbstractions.synchronize(backend)
    println("Results: ", Array(dx))
end

test_simple()
