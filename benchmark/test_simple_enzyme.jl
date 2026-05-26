using CUDA, Enzyme, KernelAbstractions

@inline function simple_f(x)
    return x * x
end

@kernel function simple_kernel!(dx, @Const(x))
    i = @index(Global)
    xi = x[i]
    # Reverse mode on scalar function with Active input
    # Returns a tuple of adjoints
    adj = Enzyme.autodiff_deferred(Reverse, Const(simple_f), Active, Active(xi))
    dx[i] = adj[1][1]
end

function test_simple()
    if !CUDA.functional(); return; end
    backend = CUDABackend()
    x = CUDA.fill(3.0, 10)
    dx = CUDA.fill(0.0, 10)
    kernel = simple_kernel!(backend)
    kernel(dx, x, ndrange=10)
    KernelAbstractions.synchronize(backend)
    println("Results: ", Array(dx))
end

test_simple()
