using CUDA, Enzyme, KernelAbstractions

@inline function simple_f_const(x, c)
    return x * x * c
end

@kernel function simple_kernel!(dx, @Const(x), c)
    i = @index(Global)
    xi = x[i]
    # Test if using only Active arguments works
    adj = Enzyme.autodiff_deferred(Reverse, Const(simple_f_const), Active, Active(xi), Active(c))
    dx[i] = adj[1]
end

function test_simple()
    if !CUDA.functional(); return; end
    backend = CUDABackend()
    x = CUDA.fill(3.0, 10)
    dx = CUDA.fill(0.0, 10)
    c = 2.0
    kernel = simple_kernel!(backend)
    kernel(dx, x, c, ndrange=10)
    KernelAbstractions.synchronize(backend)
    println("Results: ", Array(dx))
end

test_simple()
