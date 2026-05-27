#!/usr/bin/env julia

using cDFT
using cDFT.Clapeyron
using Printf
using LinearAlgebra
using BenchmarkTools
using CUDA
using Statistics

NGRIDS = 2 .^ [2, 5, 8, 11, 14]

function main()
    T = 300.0
    p = 1.0e5
    components = ["hexane"]
    model = PCSAFT(components)

    vl = volume(model, p, T)
    ρbulk = [1.0 / vl]

    L = cDFT.length_scale(model)

    println("Length scale L = $(L) m")
    println("Bulk density = $(ρbulk[1]) mol/m^3")

    μ_bulk = Clapeyron.VT_chemical_potential_res(
        model,
        1 / sum(ρbulk),
        T,
        ρbulk / sum(ρbulk),
    ) / Clapeyron.R̄ / T

    println("Bulk μ_res/RT from Clapeyron = $(μ_bulk[1])")
    println()

    # 一共 14 个字段
    @printf("%8s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s\n",
        "NGRID",
        "old_cpu_min",
        "old_cpu_med",
        "old_gpu_min",
        "old_gpu_med",
        "new_cpu_min",
        "new_cpu_med",
        "new_gpu_min",
        "new_gpu_med",
        "new_cpu/old",
        "new_gpu/old",
        "gpu/cpu_new",
        "diff_cpu",
        "diff_gpu",
    )
    println("-"^175)

    for NGRID in NGRIDS
        GC.gc()
        CUDA.reclaim()

        # ============================================================
        # CPU original old! version
        # ============================================================
        structure_cpu = Uniform1DCart((p, T), ρbulk, [-10L, 10L], NGRID)
        options_cpu = DFTOptions(CPU())
        system_cpu = DFTSystem(model, structure_cpu, options_cpu)

        ρ0_cpu = cDFT.initialize_profiles(system_cpu)

        δfδρ_old_cpu, cache_cpu, _, _ = cDFT.preallocate(system_cpu, ρ0_cpu)

        cDFT.δFδρ_res!(
            system_cpu, ρ0_cpu, δfδρ_old_cpu, cache_cpu...
        )

        bench_old_cpu = @benchmark cDFT.δFδρ_res!(
            $system_cpu, $ρ0_cpu, $δfδρ_old_cpu, $(cache_cpu)...
        )

        # ============================================================
        # CPU newautodiff! version
        # ============================================================
        δfδρ_new_cpu, cache_new_cpu, _, _ = cDFT.preallocate_newautodiff(system_cpu, ρ0_cpu)

        cDFT.δFδρ_res_newautodiff!(
            system_cpu, ρ0_cpu, δfδρ_new_cpu, cache_new_cpu...
        )

        bench_new_cpu = @benchmark cDFT.δFδρ_res_newautodiff!(
            $system_cpu, $ρ0_cpu, $δfδρ_new_cpu, $(cache_new_cpu)...
        )

        # ============================================================
        # GPU old! version
        # ============================================================
        structure_gpu = Uniform1DCart((p, T), ρbulk, [-10L, 10L], NGRID)
        options_gpu = DFTOptions(CUDABackend())
        system_gpu = DFTSystem(model, structure_gpu, options_gpu)

        ρ0_gpu = cDFT.initialize_profiles(system_gpu)

        δfδρ_old_gpu, cache_old_gpu, _, _ = cDFT.preallocate(system_gpu, ρ0_gpu)

        CUDA.@sync cDFT.δFδρ_res!(
            system_gpu, ρ0_gpu, δfδρ_old_gpu, cache_old_gpu...
        )

        bench_old_gpu = @benchmark CUDA.@sync cDFT.δFδρ_res!(
            $system_gpu, $ρ0_gpu, $δfδρ_old_gpu, $(cache_old_gpu)...
        )

        # ============================================================
        # GPU newautodiff! version
        # ============================================================
        δfδρ_new_gpu, cache_new_gpu, _, _ = cDFT.preallocate_newautodiff(system_gpu, ρ0_gpu)

        CUDA.@sync cDFT.δFδρ_res_newautodiff!(
            system_gpu, ρ0_gpu, δfδρ_new_gpu, cache_new_gpu...
        )

        bench_new_gpu = @benchmark CUDA.@sync cDFT.δFδρ_res_newautodiff!(
            $system_gpu, $ρ0_gpu, $δfδρ_new_gpu, $(cache_new_gpu)...
        )

        CUDA.synchronize()

        # ============================================================
        # Correctness comparison
        # ============================================================
        old_cpu_arr = Array(δfδρ_old_cpu)
        old_gpu_arr = Array(δfδρ_old_gpu)
        new_cpu_arr = Array(δfδρ_new_cpu)
        new_gpu_arr = Array(δfδρ_new_gpu)

        diff_new_cpu = maximum(abs.(old_cpu_arr .- new_cpu_arr))
        diff_new_gpu = maximum(abs.(old_gpu_arr .- new_gpu_arr))

        # ============================================================
        # Timing
        # ============================================================
        old_cpu_min_ms = minimum(bench_old_cpu).time / 1e6
        old_cpu_med_ms = median(bench_old_cpu).time / 1e6

        old_gpu_min_ms = minimum(bench_old_gpu).time / 1e6
        old_gpu_med_ms = median(bench_old_gpu).time / 1e6

        new_cpu_min_ms = minimum(bench_new_cpu).time / 1e6
        new_cpu_med_ms = median(bench_new_cpu).time / 1e6

        new_gpu_min_ms = minimum(bench_new_gpu).time / 1e6
        new_gpu_med_ms = median(bench_new_gpu).time / 1e6

        speedup_cpu = old_cpu_med_ms / new_cpu_med_ms
        speedup_gpu = old_gpu_med_ms / new_gpu_med_ms
        speedup_new_gpu_cpu = new_cpu_med_ms / new_gpu_med_ms

        @printf("%8d  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.3f  %12.3f  %12.3f  %12.4e  %12.4e\n",
            NGRID,
            old_cpu_min_ms,
            old_cpu_med_ms,
            old_gpu_min_ms,
            old_gpu_med_ms,
            new_cpu_min_ms,
            new_cpu_med_ms,
            new_gpu_min_ms,
            new_gpu_med_ms,
            speedup_cpu,
            speedup_gpu,
            speedup_new_gpu_cpu,
            diff_new_cpu,
            diff_new_gpu
        )
    end
end

main()