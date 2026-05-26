using CUDA, Enzyme, KernelAbstractions

@inline function simple_f_tup(tup::NTuple{2, Float64})
    return tup[1] * tup[2]
end

@kernel function simple_kernel!(dx, @Const(x), @Const(y))
    i = @index(Global)
    tup = (x[i], y[i])
    # Test if exactly ONE Active argument works when it is an NTuple
    adj = Enzyme.autodiff_deferred(Reverse, Const(simple_f_tup), Active, Active(tup))
    # adj[1] should be an NTuple{2, Float64} containing the gradients
    res_adj = adj[1]
    dx[i] = res_adj[1]
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
