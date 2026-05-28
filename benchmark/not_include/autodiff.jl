# bench_rosenbrock_enzyme.jl
#
# Run:
#   julia --project bench_rosenbrock_enzyme.jl
#
# Install if needed:
#   import Pkg
#   Pkg.add(["CUDA", "Enzyme", "BenchmarkTools", "Statistics"])

using CUDA
using Enzyme
using BenchmarkTools
using Statistics

# -----------------------------
# Rosenbrock scalar function
# f(x, y) = (a - x)^2 + b(y - x^2)^2
# -----------------------------
@inline function rosenbrock2(x::T, y::T) where {T}
    a = T(1)
    b = T(100)
    return (a - x)^2 + b * (y - x^2)^2
end

# Analytic gradient for correctness check
@inline function rosenbrock2_grad_analytic(x::T, y::T) where {T}
    gx = -2 * (T(1) - x) - T(400) * x * (y - x^2)
    gy = T(200) * (y - x^2)
    return gx, gy
end

# -----------------------------
# CPU autodiff kernel:
# loop over many independent Rosenbrock problems
# -----------------------------
function cpu_enzyme_grad!(gx, gy, xs, ys)
    @inbounds for i in eachindex(xs)
        # Reverse-mode Enzyme for scalar-valued function.
        # Output is Active, inputs are Active.
        res = Enzyme.autodiff(
            Enzyme.Reverse,
            rosenbrock2,
            Enzyme.Active,
            Enzyme.Active(xs[i]),
            Enzyme.Active(ys[i]),
        )

        # res[1] is the tuple of input adjoints.
        gx[i] = res[1][1]
        gy[i] = res[1][2]
    end
    return nothing
end

# -----------------------------
# GPU primal kernel body
# Computes f[i] = Rosenbrock(xs[i], ys[i])
# -----------------------------
function gpu_rosenbrock_primal!(f, xs, ys, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= n
        x = xs[i]
        y = ys[i]
        f[i] = rosenbrock2(x, y)
    end

    return nothing
end

# -----------------------------
# GPU Enzyme gradient kernel
# Differentiates gpu_rosenbrock_primal!
# -----------------------------
function gpu_enzyme_grad_kernel!(f, df, xs, dxs, ys, dys, n)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Const(gpu_rosenbrock_primal!),
        Const,
        Duplicated(f, df),
        Duplicated(xs, dxs),
        Duplicated(ys, dys),
        Const(n),
    )

    return nothing
end

function gpu_enzyme_grad!(f_d, df_d, xs_d, dxs_d, ys_d, dys_d)
    n = length(xs_d)
    threads = 256
    blocks = cld(n, threads)

    CUDA.fill!(f_d, zero(eltype(f_d)))
    CUDA.fill!(dxs_d, zero(eltype(dxs_d)))
    CUDA.fill!(dys_d, zero(eltype(dys_d)))

    # Seed d(sum(f))/df = 1 for every element.
    CUDA.fill!(df_d, one(eltype(df_d)))

    @cuda threads=threads blocks=blocks gpu_enzyme_grad_kernel!(
        f_d, df_d, xs_d, dxs_d, ys_d, dys_d, n
    )

    return nothing
end

# -----------------------------
# Optional non-AD GPU analytic kernel
# Useful as a performance reference
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
    end
    println("N = $N, T = $T")

    # Generate inputs near Rosenbrock valley, but not exactly at optimum.
    xs = rand(T, N) .* T(4) .- T(2)      # [-2, 2]
    ys = rand(T, N) .* T(6) .- T(1)      # [-1, 5]

    gx_cpu = similar(xs)
    gy_cpu = similar(ys)

    # Warmup CPU compilation
    cpu_enzyme_grad!(gx_cpu, gy_cpu, xs, ys)

    # Correctness against analytic gradient
    gx_ref = similar(xs)
    gy_ref = similar(ys)

    @inbounds for i in eachindex(xs)
        gx_ref[i], gy_ref[i] = rosenbrock2_grad_analytic(xs[i], ys[i])
    end

    println()
    println("CPU Enzyme correctness:")
    println("  max |gx - gx_ref| = ", maximum(abs.(gx_cpu .- gx_ref)))
    println("  max |gy - gy_ref| = ", maximum(abs.(gy_cpu .- gy_ref)))

    println()
    println("CPU Enzyme benchmark:")
    cpu_trial = @benchmark cpu_enzyme_grad!($gx_cpu, $gy_cpu, $xs, $ys)
    display(cpu_trial)

    if !CUDA.functional()
        println("CUDA is not functional on this machine; skipping GPU benchmark.")
        return
    end

    xs_d = CuArray(xs)
    ys_d = CuArray(ys)

    f_d   = CUDA.zeros(T, N)
    df_d  = CUDA.ones(T, N)
    dxs_d = CUDA.zeros(T, N)
    dys_d = CUDA.zeros(T, N)
    gx_d  = CUDA.zeros(T, N)
    gy_d  = CUDA.zeros(T, N)

    # Warmup GPU Enzyme
    CUDA.@sync gpu_enzyme_grad!(f_d, df_d, xs_d, dxs_d, ys_d, dys_d)

    gx_gpu = Array(dxs_d)
    gy_gpu = Array(dys_d)

    println()
    println("GPU Enzyme correctness:")
    println("  max |gx - gx_ref| = ", maximum(abs.(gx_gpu .- gx_ref)))
    println("  max |gy - gy_ref| = ", maximum(abs.(gy_gpu .- gy_ref)))

    println()
    println("GPU Enzyme benchmark:")
    gpu_trial = @benchmark CUDA.@sync gpu_enzyme_grad!(
        $f_d, $df_d, $xs_d, $dxs_d, $ys_d, $dys_d
    )
    display(gpu_trial)

    # Optional analytic GPU reference
    CUDA.@sync gpu_analytic_grad!(gx_d, gy_d, xs_d, ys_d)

    println()
    println("GPU analytic gradient benchmark, no autodiff:")
    gpu_analytic_trial = @benchmark CUDA.@sync gpu_analytic_grad!($gx_d, $gy_d, $xs_d, $ys_d)
    display(gpu_analytic_trial)

    println()
    println("Summary, median times:")
    println("  CPU Enzyme:       ", median(cpu_trial).time / 1e6, " ms")
    println("  GPU Enzyme:       ", median(gpu_trial).time / 1e6, " ms")
    println("  GPU analytic:     ", median(gpu_analytic_trial).time / 1e6, " ms")
    println("  CPU/GPU AD speedup: ",
            median(cpu_trial).time / median(gpu_trial).time, "x")
end

main()