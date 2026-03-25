"""
    benchmark_fft.jl

Compare R2C (real-to-complex) FFT throughput between CPU (FFTW) and Apple Metal GPU
across grid sizes and dimensionalities — the core operation of the SCFT propagator.

Each timed "round-trip" is:  forward R2C FFT → element-wise kernel multiply → inverse C2R FFT.
This mirrors exactly what `convolve!` / `batched_convolve_pair!` do per propagation step.

Different batch sizes show how performance scales when multiple planes are packed into
a single FFT call (SCFT currently uses batch=2 on Metal; larger batches may help).

Run:
    julia --project examples/benchmark_fft.jl
    julia --project -t 8 examples/benchmark_fft.jl   # with 8 CPU threads
"""

using FFTW
using Printf
using LinearAlgebra
using Metal

# ─── Timing helper ────────────────────────────────────────────────────────────
"""
Time one round-trip (forward R2C → kernel multiply → inverse C2R), averaged over
`nruns` iterations after `warmup` warm-up calls.

`sync!()` must block until all pending GPU work finishes; pass `Metal.synchronize`
for Metal or `() -> nothing` for CPU.
"""
function bench_convolve!(buf_r, buf_c, kernel, P, iP, sync!;
                         warmup=10, nruns=100)
    for _ in 1:warmup
        mul!(buf_c, P, buf_r)
        buf_c .*= kernel
        mul!(buf_r, iP, buf_c)
    end
    sync!()

    t = @elapsed begin
        for _ in 1:nruns
            mul!(buf_c, P, buf_r)
            buf_c .*= kernel
            mul!(buf_r, iP, buf_c)
        end
        sync!()
    end
    return t / nruns   # seconds per round-trip
end

function print_section(title)
    println()
    println("── " * title * " " * "─"^max(0, 58 - length(title)))
end

# ─── System info ──────────────────────────────────────────────────────────────
println("=" ^ 62)
println("R2C FFT Round-Trip Benchmark")
println("Julia threads available: $(Threads.nthreads())")
if Metal.functional()
    println("Metal GPU: $(Metal.device().name)")
    let d = Metal.device()
        avail = (d.recommendedMaxWorkingSetSize - d.currentAllocatedSize) / 1024^3
        @printf("VRAM available: %.2f GB\n", avail)
    end
end
println("=" ^ 62)
FFTW.set_num_threads(Threads.nthreads())

all_results = Dict{String, Float64}()   # (grid_label, backend, batch) => time_s

# ─── Benchmark configurations ─────────────────────────────────────────────────
# Each entry: (label, dims, ndim)  where dims gives the spatial grid
configs = [
    ("2D 512×512",   (512, 512),       2),
    ("2D 128×128",   (128, 128),       2),
    ("3D  64× 64×64",  (64,  64,  64), 3),
    ("3D 128×128×128", (128, 128, 128),3),
]

for (glabel, gdims, ndim) in configs
    Nx       = gdims[1]
    rfft_sz  = (Nx ÷ 2 + 1, gdims[2:end]...)
    fft_axes = 1:ndim
    nelems   = prod(gdims)

    print_section("$(glabel)  ($(nelems) real elements/plane)")

    # ── CPU ──────────────────────────────────────────────────────────────────
    for (FT, flabel) in ((Float64, "Float64"), (Float32, "Float32"))
        CT = Complex{FT}
        for batch in (1, 2, 4)
            buf_r  = rand(FT, gdims..., batch)
            buf_c  = zeros(CT, rfft_sz..., batch)
            kernel = rand(CT, rfft_sz..., batch)

            P  = plan_rfft(buf_r,  fft_axes; num_threads=Threads.nthreads())
            iP = plan_irfft(buf_c, Nx, fft_axes; num_threads=Threads.nthreads())

            t = bench_convolve!(buf_r, buf_c, kernel, P, iP, () -> nothing)
            @printf("  CPU %-7s  batch=%-2d  %8.3f ms\n", flabel, batch, t * 1e3)
            all_results["$(glabel)|cpu_$(flabel)|b$(batch)"] = t
        end
    end

    # ── Metal ────────────────────────────────────────────────────────────────
    if Metal.functional()
        FT = Float32
        CT = Complex{FT}
        println()
        for batch in (1, 2, 4, 8)
            buf_r_gpu  = MtlArray(rand(FT, gdims..., batch))
            buf_c_gpu  = MtlArray(zeros(CT, rfft_sz..., batch))
            kernel_gpu = MtlArray(rand(CT, rfft_sz..., batch))

            P  = plan_rfft(buf_r_gpu,  fft_axes)
            iP = plan_irfft(buf_c_gpu, Nx, fft_axes)

            t = bench_convolve!(buf_r_gpu, buf_c_gpu, kernel_gpu, P, iP, Metal.synchronize)
            @printf("  Metal Float32  batch=%-2d  %8.3f ms\n", batch, t * 1e3)
            all_results["$(glabel)|metal|b$(batch)"] = t
        end

        # speedup ratios for this grid
        println()
        @printf("  %-8s  %-22s  %-22s\n", "", "CPU-Float64 / Metal", "CPU-Float32 / Metal")
        for batch in (1, 2, 4)
            t64  = get(all_results, "$(glabel)|cpu_Float64|b$(batch)", NaN)
            t32  = get(all_results, "$(glabel)|cpu_Float32|b$(batch)", NaN)
            tgpu = get(all_results, "$(glabel)|metal|b$(batch)",       NaN)
            @printf("  batch=%-2d    %8.2f×                  %8.2f×\n",
                    batch, t64/tgpu, t32/tgpu)
        end
    end
end

println()
println("── Legend " * "─"^51)
println("  Round-trip = forward R2C + element-wise kernel multiply + inverse C2R")
println("  batch=N:  N spatial planes packed into one FFT call")
println("  ratio > 1.0 → Metal is faster  |  ratio < 1.0 → CPU is faster")
println("  SCFT uses Float64 on CPU and Float32 on Metal (batch=2 currently).")
println()
println("── Interpretation ─────────────────────────────────────────")
println("  If Metal time is FLAT across batch sizes → per-mul! overhead dominates")
println("    (reducing total mul! calls will help, not increasing batch size)")
println("  If Metal time GROWS with batch → compute-bound, larger batch hurts")
println("  If Metal time is similar to CPU → FFT not the bottleneck")
