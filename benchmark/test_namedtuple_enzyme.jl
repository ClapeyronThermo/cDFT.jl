using CUDA, Enzyme, KernelAbstractions

@inline function f_tup(args)
    return args.x * args.y
end

@kernel function simple_kernel!(dx, @Const(x), @Const(y))
    i = @index(Global)
    args = (x = x[i], y = y[i])
    # Test if ONE Active argument as NamedTuple works
    adj = Enzyme.autodiff_deferred(Reverse, Const(f_tup), Active, Active(args))
    dx[i] = adj[1].x
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
