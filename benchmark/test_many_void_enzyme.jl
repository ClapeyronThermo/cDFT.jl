using CUDA, Enzyme, KernelAbstractions

@inline function f_many_void(out, x, i, a, b, c, d, e, f, g)
    out[i] = x[i] + a + b + c + d + e + f + g
    return nothing
end

@kernel function grad_kernel!(dx, x, out, dout)
    i = @index(Global)
    Enzyme.autodiff_deferred(Reverse, Const(f_many_void), Const,
        Duplicated(out, dout),
        Duplicated(x, dx),
        Const(i),
        Const(1.0), Const(1.0), Const(1.0), Const(1.0), Const(1.0), Const(1.0), Const(1.0))
end

function test()
    if !CUDA.functional(); return; end
    backend = CUDABackend()
    N = 10
    x = CUDA.fill(3.0, N)
    out = CUDA.fill(0.0, N)
    dx = CUDA.fill(0.0, N)
    dout = CUDA.fill(1.0, N)
    
    kernel = grad_kernel!(backend)
    kernel(dx, x, out, dout, ndrange=N)
    KernelAbstractions.synchronize(backend)
    println("Results: ", Array(dx))
end

test()
