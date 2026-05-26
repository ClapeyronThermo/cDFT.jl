using CUDA, Enzyme, KernelAbstractions

@inline function f_3d_void(out, x, i)
    out[i] = x[1, 1, i]
    return nothing
end

@kernel function grad_kernel!(dx, x, out, dout)
    i = @index(Global)
    Enzyme.autodiff_deferred(Reverse, Const(f_3d_void), Const,
        Duplicated(out, dout),
        Duplicated(x, dx),
        Const(i))
end

function test()
    if !CUDA.functional(); return; end
    backend = CUDABackend()
    N = 10
    x = CUDA.fill(3.0, 1, 1, N)
    out = CUDA.fill(0.0, N)
    dx = CUDA.fill(0.0, 1, 1, N)
    dout = CUDA.fill(1.0, N)
    
    kernel = grad_kernel!(backend)
    kernel(dx, x, out, dout, ndrange=N)
    KernelAbstractions.synchronize(backend)
    println("Results: ", Array(dx)[1, 1, :])
end

test()
