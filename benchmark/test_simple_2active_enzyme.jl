using CUDA, Enzyme, KernelAbstractions

@inline function simple_f_2(x, y)
    return x * y
end

@kernel function simple_kernel!(dx, @Const(x), @Const(y))
    i = @index(Global)
    xi = x[i]
    yi = y[i]
    # Use two Active arguments
    adj = Enzyme.autodiff_deferred(Reverse, Const(simple_f_2), Active, Active(xi), Active(yi))
    dx[i] = adj[1]
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
