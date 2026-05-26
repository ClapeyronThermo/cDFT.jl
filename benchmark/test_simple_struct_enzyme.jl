using CUDA, Enzyme, KernelAbstractions

struct MyArgs
    x::Float64
    y::Float64
end

@inline function simple_f_struct(args::MyArgs)
    return args.x * args.y
end

@kernel function simple_kernel!(dx, @Const(x), @Const(y))
    i = @index(Global)
    args = MyArgs(x[i], y[i])
    # Test if exactly ONE Active argument works when it is a custom immutable struct
    adj = Enzyme.autodiff_deferred(Reverse, Const(simple_f_struct), Active, Active(args))
    # adj[1] should be a MyArgs struct containing the gradients
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
