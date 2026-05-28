using CUDA
using ForwardDiff
using BenchmarkTools
using Statistics

# -----------------------------
# Rosenbrock scalar function
# f(x, y) = (1 - x)^2 + 100(y - x^2)^2
# -----------------------------
@inline function rosenbrock2(x, y)
    return (one(x) - x)^2 + oftype(x, 100) * (y - x^2)^2
end

@inline function rosenbrock2_grad_analytic(x::T, y::T) where {T}
    gx = -2 * (one(T) - x) - T(400) * x * (y - x^2)
    gy = T(200) * (y - x^2)
    return gx, gy
end

# -----------------------------
# CPU ForwardDiff benchmark
# -----------------------------
function cpu_forwarddiff_grad!(gx, gy, xs, ys)
    @inbounds for i in eachindex(xs)
        x = xs[i]
        y = ys[i]

        xd = ForwardDiff.Dual{Nothing}(x, one(x), zero(x))
        yd = ForwardDiff.Dual{Nothing}(y, zero(y), one(y))

        fd = rosenbrock2(xd, yd)

        gx[i] = ForwardDiff.partials(fd, 1)
        gy[i] = ForwardDiff.partials(fd, 2)
    end
    return nothing
end

# -----------------------------
# GPU ForwardDiff kernel
# One CUDA thread computes gradient for one (x, y)
# -----------------------------
function gpu_forwarddiff_grad_kernel!(gx, gy, xs, ys, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= n
        x = xs[i]
        y = ys[i]

        xd = ForwardDiff.Dual{Nothing}(x, one(x), zero(x))
        yd = ForwardDiff.Dual{Nothing}(y, zero(y), one(y))

        fd = rosenbrock2(xd, yd)

        gx[i] = ForwardDiff.partials(fd, 1)
        gy[i] = ForwardDiff.partials(fd, 2)
    end

    return nothing
end

function gpu_forwarddiff_grad!(gx_d, gy_d, xs_d, ys_d)
    n = length(xs_d)
    threads = 256
    blocks = cld(n, threads)

    @cuda threads=threads blocks=blocks gpu_forwarddiff_grad_kernel!(
        gx_d, gy_d, xs_d, ys_d, n
    )

    return nothing
end

# -----------------------------
# GPU analytic kernel
# Reference performance lower bound
# -----------------------------
function gpu_analytic_grad_kernel!(gx, gy, xs, ys, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= n
        gx_i, gy_i = rosenbrock2_grad_analytic(xs[i], ys[i])
        gx[i] = gx_i
        gy[i] = gy_i
    end

    return nothing
end

function gpu_analytic_grad!(gx_d, gy_d, xs_d, ys_d)
    n = length(xs_d)
    threads = 256
    blocks = cld(n, threads)

    @cuda threads=threads blocks=blocks gpu_analytic_grad_kernel!(
        gx_d, gy_d, xs_d, ys_d, n
    )

    return nothing
end

# -----------------------------
# Benchmark driver
# -----------------------------
function main(; N = 1_000_000, T = Float32)
    println("Julia version: ", VERSION)
    println("CUDA functional: ", CUDA.functional())

    if CUDA.functional()
        println("CUDA device: ", CUDA.name(CUDA.device()))
    else
        error("CUDA is not functional on this machine.")
    end

    println("N = $N, T = $T")

    xs = rand(T, N) .* T(4) .- T(2)
    ys = rand(T, N) .* T(6) .- T(1)

    gx_ref = similar(xs)
    gy_ref = similar(ys)

    @inbounds for i in eachindex(xs)
        gx_ref[i], gy_ref[i] = rosenbrock2_grad_analytic(xs[i], ys[i])
    end

    # -----------------------------
    # CPU ForwardDiff
    # -----------------------------
    gx_cpu = similar(xs)
    gy_cpu = similar(ys)

    cpu_forwarddiff_grad!(gx_cpu, gy_cpu, xs, ys)

    println()
    println("CPU ForwardDiff correctness:")
    println("  max |gx - gx_ref| = ", maximum(abs.(gx_cpu .- gx_ref)))
    println("  max |gy - gy_ref| = ", maximum(abs.(gy_cpu .- gy_ref)))

    println()
    println("CPU ForwardDiff benchmark:")
    cpu_fd_trial = @benchmark cpu_forwarddiff_grad!($gx_cpu, $gy_cpu, $xs, $ys)
    display(cpu_fd_trial)

    # -----------------------------
    # GPU setup
    # -----------------------------
    xs_d = CuArray(xs)
    ys_d = CuArray(ys)

    gx_d = CUDA.zeros(T, N)
    gy_d = CUDA.zeros(T, N)

    # -----------------------------
    # GPU ForwardDiff
    # -----------------------------
    CUDA.@sync gpu_forwarddiff_grad!(gx_d, gy_d, xs_d, ys_d)

    gx_gpu_fd = Array(gx_d)
    gy_gpu_fd = Array(gy_d)

    println()
    println("GPU ForwardDiff correctness:")
    println("  max |gx - gx_ref| = ", maximum(abs.(gx_gpu_fd .- gx_ref)))
    println("  max |gy - gy_ref| = ", maximum(abs.(gy_gpu_fd .- gy_ref)))

    println()
    println("GPU ForwardDiff benchmark:")
    gpu_fd_trial = @benchmark CUDA.@sync gpu_forwarddiff_grad!(
        $gx_d, $gy_d, $xs_d, $ys_d
    )
    display(gpu_fd_trial)

    # -----------------------------
    # GPU analytic reference
    # -----------------------------
    CUDA.@sync gpu_analytic_grad!(gx_d, gy_d, xs_d, ys_d)

    gx_gpu_an = Array(gx_d)
    gy_gpu_an = Array(gy_d)

    println()
    println("GPU analytic correctness:")
    println("  max |gx - gx_ref| = ", maximum(abs.(gx_gpu_an .- gx_ref)))
    println("  max |gy - gy_ref| = ", maximum(abs.(gy_gpu_an .- gy_ref)))

    println()
    println("GPU analytic benchmark:")
    gpu_an_trial = @benchmark CUDA.@sync gpu_analytic_grad!(
        $gx_d, $gy_d, $xs_d, $ys_d
    )
    display(gpu_an_trial)

    println()
    println("Summary, median times:")
    println("  CPU ForwardDiff:  ", median(cpu_fd_trial).time / 1e6, " ms")
    println("  GPU ForwardDiff:  ", median(gpu_fd_trial).time / 1e6, " ms")
    println("  GPU analytic:     ", median(gpu_an_trial).time / 1e6, " ms")
    println("  CPU/GPU FD speedup: ",
            median(cpu_fd_trial).time / median(gpu_fd_trial).time, "x")
    println("  GPU FD / analytic overhead: ",
            median(gpu_fd_trial).time / median(gpu_an_trial).time, "x")
end

main()