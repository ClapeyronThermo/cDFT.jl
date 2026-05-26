using CUDA, Enzyme, KernelAbstractions

@inline function f_binary(x, y)
    return x * y
end

@kernel function kernel!(dx, @Const(x), @Const(y))
    i = @index(Global)
    xi = x[i]
    yi = y[i]
    # Try 2 active arguments
    adj = Enzyme.autodiff_deferred(Reverse, Const(f_binary), Active, Active(xi), Active(yi))
    # If it works, adj is a Tuple(Float64, Float64)
    dx[i] = adj[1]
end

function test()
    if !CUDA.functional(); return; end
    backend = CUDABackend()
    x = CUDA.fill(3.0, 10)
    y = CUDA.fill(2.0, 10)
    dx = CUDA.fill(0.0, 10)
    kernel = kernel!(backend)
    kernel(dx, x, y, ndrange=10)
    KernelAbstractions.synchronize(backend)
    println("Results: ", Array(dx))
end

test()
