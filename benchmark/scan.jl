#!/usr/bin/env julia

using cDFT
using cDFT.Clapeyron
using Printf
using LinearAlgebra
using BenchmarkTools
using CUDA
using Statistics

NGRIDS = 2 .^ [6, 9, 12, 15, 18, 20, 21, 22]

function main()
    T = 300.0
    p = 1.0e5
    components = ["hexane"]
    model = PCSAFT(components)

    vl = Clapeyron.volume(model, p, T)
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

    f = open("scan.txt", "w")
    @printf("%8s  %12s  %12s  %12s  %12s  %16s  %16s  %16s\n",
        "NGRID",
        "old_cpu",
        "old_gpu",
        "new_cpu",
        "new_gpu",
        "SU:new/old_c",
        "SU:new/old_g",
        "SU:new_g/c"
    )
    @printf(f, "%8s  %12s  %12s  %12s  %12s  %16s  %16s  %16s\n",
        "NGRID",
        "old_cpu",
        "old_gpu",
        "new_cpu",
        "new_gpu",
        "SU:new/old_c",
        "SU:new/old_g",
        "SU:new_g/c"
    )
    line = "-"^110
    println(line)
    println(f, line)

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

        δfδρ_new_cpu, cache_new_cpu, _, _ =
            cDFT.preallocate_newautodiff(system_cpu, ρ0_cpu)

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

        CUDA.synchronize()

        bench_old_gpu = @benchmark CUDA.@sync cDFT.δFδρ_res!(
            $system_gpu, $ρ0_gpu, $δfδρ_old_gpu, $(cache_old_gpu)...
        )
        CUDA.synchronize()

        # ============================================================
        # GPU newautodiff! version
        # ============================================================
        δfδρ_new_gpu, cache_new_gpu, _, _ =
            cDFT.preallocate_newautodiff(system_gpu, ρ0_gpu)

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
        if diff_new_cpu > 1e-6 || diff_new_gpu > 1e-6
            @printf("ERROR: Maximum absolute difference between old and new results exceeds tolerance! diff_new_cpu = %e, diff_new_gpu = %e\n", diff_new_cpu, diff_new_gpu)
        end
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

        @printf("%8d  %12.6f  %12.6f  %12.6f  %12.6f  %16.6f  %16.6f  %16.6f\n",
            NGRID,
            old_cpu_med_ms,
            old_gpu_med_ms,
            new_cpu_med_ms,
            new_gpu_med_ms,
            speedup_cpu,
            speedup_gpu,
            speedup_new_gpu_cpu
        )
        @printf(f, "%8d  %12.6f  %12.6f  %12.6f  %12.6f  %16.6f  %16.6f  %16.6f\n",
            NGRID,
            old_cpu_med_ms,
            old_gpu_med_ms,
            new_cpu_med_ms,
            new_gpu_med_ms,
            speedup_cpu,
            speedup_gpu,
            speedup_new_gpu_cpu
        )
        flush(f)
    end
    close(f)
end

main()