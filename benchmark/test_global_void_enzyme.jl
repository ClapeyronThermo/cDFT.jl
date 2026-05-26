using CUDA, Enzyme, KernelAbstractions

# Primal device function operating on global memory at an index
@inline function simple_f_void(out, x, y, i)
    out[i] = x[i] * y[i]
    return nothing
end

@kernel function grad_kernel!(dx, dy, x, y, out, dout)
    i = @index(Global)
    # Use autodiff_deferred on the void function with global arrays
    Enzyme.autodiff_deferred(Reverse, Const(simple_f_void), Const,
        Duplicated(out, dout),
        Duplicated(x, dx),
        Duplicated(y, dy),
        Const(i))
end

function test()
    if !CUDA.functional(); return; end
    backend = CUDABackend()
    N = 10
    x = CUDA.fill(3.0, N)
    y = CUDA.fill(2.0, N)
    out = CUDA.fill(0.0, N)
    
    dx = CUDA.fill(0.0, N)
    dy = CUDA.fill(0.0, N)
    dout = CUDA.fill(1.0, N) # Seed gradient
    
    kernel = grad_kernel!(backend)
    kernel(dx, dy, x, y, out, dout, ndrange=N)
    KernelAbstractions.synchronize(backend)
    
    println("dx: ", Array(dx))
    println("dy: ", Array(dy))
end

test()
