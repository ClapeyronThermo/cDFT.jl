using CUDA, Enzyme, KernelAbstractions

@inline function f_val_void(out, x, i, ::Val{N}) where N
    out[i] = x[i] * N
    return nothing
end

@kernel function grad_kernel!(dx, x, out, dout)
    i = @index(Global)
    Enzyme.autodiff_deferred(Reverse, Const(f_val_void), Const,
        Duplicated(out, dout),
        Duplicated(x, dx),
        Const(i),
        Const(Val(2)))
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
