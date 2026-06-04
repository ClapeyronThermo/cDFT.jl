#!/usr/bin/env julia

using cDFT
using cDFT.Clapeyron
using Printf
using LinearAlgebra
using BenchmarkTools
using CUDA
using Statistics

# For 3D, use reasonable grid sizes.
# 51^3 already has 132651 grid points.
NGRIDS = [16, 32]

# ============================================================
# Memory helper
# ============================================================
function bytes_human(n)
    if n < 0
        return "-" * bytes_human(-n)
    elseif n < 1024
        return @sprintf("%.2fB", n)
    elseif n < 1024^2
        return @sprintf("%.2fKB", n / 1024)
    elseif n < 1024^3
        return @sprintf("%.2fMB", n / 1024^2)
    else
        return @sprintf("%.2fGB", n / 1024^3)
    end
end

# ============================================================
# Build acetone/hexane LLE mixture model
# ============================================================
function build_mixture_system_info()
    p = 1.0e5
    T = 290.15

    model = PCSAFT(["acetone", "hexane"])

    eps11 = model.params.epsilon.values[1, 1]
    eps22 = model.params.epsilon.values[2, 2]
    eps12 = sqrt(eps11 * eps22) * 0.7

    model.params.epsilon.values[1, 2] = eps12
    model.params.epsilon.values[2, 1] = eps12

    x, n_flash, G = tp_flash(
        model,
        p,
        T,
        [0.5, 0.5],
        MichelsenTPFlash(equilibrium = :lle, K0 = [100.0, 0.001]),
    )

    v1 = volume(model, p, T, x[1, :])
    v2 = volume(model, p, T, x[2, :])

    ρ1 = x[1, :] ./ v1
    ρ2 = x[2, :] ./ v2

    ρbulk = (ρ1 .+ ρ2) ./ 2

    L = cDFT.length_scale(model)

    return model, p, T, x, ρ1, ρ2, ρbulk, L
end

function make_structure(p, T, ρbulk, L, NGRID)
    return cDFT.Uniform3DCart(
        (p, T),
        ρbulk,
        [-10L 10L;
         -10L 10L;
         -10L 10L],
        (NGRID, NGRID, NGRID),
    )
end

# ============================================================
# CUDA allocation helper functions
# ============================================================
function measure_cuda_preallocate_old(system_gpu, ρ0_gpu)
    out = Ref{Any}()

    mem = CUDA.@allocated begin
        out[] = cDFT.preallocate(system_gpu, ρ0_gpu)
        CUDA.synchronize()
    end

    return mem, out[]
end

function measure_cuda_preallocate_new(system_gpu, ρ0_gpu)
    out = Ref{Any}()

    mem = CUDA.@allocated begin
        out[] = cDFT.preallocate_newautodiff(system_gpu, ρ0_gpu)
        CUDA.synchronize()
    end

    return mem, out[]
end

function measure_cuda_run_old(system_gpu, ρ0_gpu, δfδρ_old_gpu, cache_old_gpu)
    mem = CUDA.@allocated begin
        CUDA.@sync cDFT.δFδρ_res!(
            system_gpu,
            ρ0_gpu,
            δfδρ_old_gpu,
            cache_old_gpu...,
        )
        CUDA.synchronize()
    end

    return mem
end

function measure_cuda_run_new(system_gpu, ρ0_gpu, δfδρ_new_gpu, cache_new_gpu)
    mem = CUDA.@allocated begin
        CUDA.@sync cDFT.δFδρ_res_newautodiff!(
            system_gpu,
            ρ0_gpu,
            δfδρ_new_gpu,
            cache_new_gpu...,
        )
        CUDA.synchronize()
    end

    return mem
end

function main()
    model, p, T, x_lle, ρ1, ρ2, ρbulk, L = build_mixture_system_info()

    println("Mixture: acetone / hexane")
    println("T = $(T) K")
    println("p = $(p) Pa")
    println("Length scale L = $(L) m")
    println()

    println("LLE mole fractions:")
    println("x phase 1 = $(x_lle[1, :])")
    println("x phase 2 = $(x_lle[2, :])")
    println()

    println("Phase densities:")
    println("ρ1 = $(ρ1) mol/m^3")
    println("ρ2 = $(ρ2) mol/m^3")
    println("ρbulk = $(ρbulk) mol/m^3")
    println()

    μ_bulk = Clapeyron.VT_chemical_potential_res(
        model,
        1 / sum(ρbulk),
        T,
        ρbulk / sum(ρbulk),
    ) / Clapeyron.R̄ / T

    println("Bulk μ_res/RT from Clapeyron = $(μ_bulk)")
    println()

    @printf(
        "%8s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %12s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %10s  %10s  %10s  %10s  %12s  %12s  %12s  %12s  %12s\n",
        "NGRID",
        "old_cpu_min",
        "old_cpu_med",
        "old_gpu_min",
        "old_gpu_med",
        "new_cpu_min",
        "new_cpu_med",
        "new_gpu_min",
        "new_gpu_med",
        "old_cpu_alloc",
        "new_cpu_alloc",
        "old_gpu_host",
        "new_gpu_host",
        "old_gpu_pre",
        "new_gpu_pre",
        "old_gpu_run",
        "new_gpu_run",
        "old_cpu_n",
        "new_cpu_n",
        "old_gpu_n",
        "new_gpu_n",
        "new_cpu/old",
        "new_gpu/old",
        "gpu/cpu_new",
        "diff_cpu",
        "diff_gpu",
    )

    println("-"^360)

    for NGRID in NGRIDS
        GC.gc()
        CUDA.reclaim()

        # ============================================================
        # CPU original old! version
        # ============================================================
        structure_cpu = make_structure(p, T, ρbulk, L, NGRID)
        options_cpu = DFTOptions(CPU())
        system_cpu = DFTSystem(model, structure_cpu, options_cpu)

        ρ0_cpu = cDFT.initialize_profiles(system_cpu)

        δfδρ_old_cpu, cache_cpu, _, _ =
            cDFT.preallocate(system_cpu, ρ0_cpu)

        # Warm-up
        cDFT.δFδρ_res!(
            system_cpu,
            ρ0_cpu,
            δfδρ_old_cpu,
            cache_cpu...,
        )

        bench_old_cpu = @benchmark cDFT.δFδρ_res!(
            $system_cpu,
            $ρ0_cpu,
            $δfδρ_old_cpu,
            $(cache_cpu)...,
        )

        old_cpu_mem = BenchmarkTools.memory(bench_old_cpu)
        old_cpu_allocs = BenchmarkTools.allocs(bench_old_cpu)

        # ============================================================
        # CPU newautodiff! version
        # ============================================================
        δfδρ_new_cpu, cache_new_cpu, _, _ =
            cDFT.preallocate_newautodiff(system_cpu, ρ0_cpu)

        # Warm-up
        cDFT.δFδρ_res_newautodiff!(
            system_cpu,
            ρ0_cpu,
            δfδρ_new_cpu,
            cache_new_cpu...,
        )

        bench_new_cpu = @benchmark cDFT.δFδρ_res_newautodiff!(
            $system_cpu,
            $ρ0_cpu,
            $δfδρ_new_cpu,
            $(cache_new_cpu)...,
        )

        new_cpu_mem = BenchmarkTools.memory(bench_new_cpu)
        new_cpu_allocs = BenchmarkTools.allocs(bench_new_cpu)

        old_cpu_arr = Array(δfδρ_old_cpu)
        new_cpu_arr = Array(δfδρ_new_cpu)
        diff_new_cpu = maximum(abs.(old_cpu_arr .- new_cpu_arr))

        # Free CPU-side large objects before GPU section
        cache_cpu = nothing
        cache_new_cpu = nothing
        δfδρ_old_cpu = nothing
        δfδρ_new_cpu = nothing
        old_cpu_arr = nothing
        new_cpu_arr = nothing
        ρ0_cpu = nothing
        system_cpu = nothing
        structure_cpu = nothing
        GC.gc()

        # ============================================================
        # GPU setup
        # ============================================================
        structure_gpu = make_structure(p, T, ρbulk, L, NGRID)
        options_gpu = DFTOptions(CUDABackend())
        system_gpu = DFTSystem(model, structure_gpu, options_gpu)

        ρ0_gpu = cDFT.initialize_profiles(system_gpu)

        GC.gc()
        CUDA.reclaim()
        CUDA.synchronize()

        # ============================================================
        # GPU old! version
        # ============================================================
        old_gpu_prealloc_mem, old_prealloc_result =
            measure_cuda_preallocate_old(system_gpu, ρ0_gpu)

        δfδρ_old_gpu, cache_old_gpu, _, _ = old_prealloc_result

        # Warm-up
        CUDA.@sync cDFT.δFδρ_res!(
            system_gpu,
            ρ0_gpu,
            δfδρ_old_gpu,
            cache_old_gpu...,
        )

        old_gpu_run_mem = measure_cuda_run_old(
            system_gpu,
            ρ0_gpu,
            δfδρ_old_gpu,
            cache_old_gpu,
        )

        bench_old_gpu = @benchmark CUDA.@sync cDFT.δFδρ_res!(
            $system_gpu,
            $ρ0_gpu,
            $δfδρ_old_gpu,
            $(cache_old_gpu)...,
        )

        old_gpu_host_mem = BenchmarkTools.memory(bench_old_gpu)
        old_gpu_host_allocs = BenchmarkTools.allocs(bench_old_gpu)

        # ============================================================
        # GPU newautodiff! version
        # ============================================================
        CUDA.synchronize()

        new_gpu_prealloc_mem, new_prealloc_result =
            measure_cuda_preallocate_new(system_gpu, ρ0_gpu)

        δfδρ_new_gpu, cache_new_gpu, _, _ = new_prealloc_result

        # Warm-up
        CUDA.@sync cDFT.δFδρ_res_newautodiff!(
            system_gpu,
            ρ0_gpu,
            δfδρ_new_gpu,
            cache_new_gpu...,
        )

        new_gpu_run_mem = measure_cuda_run_new(
            system_gpu,
            ρ0_gpu,
            δfδρ_new_gpu,
            cache_new_gpu,
        )

        bench_new_gpu = @benchmark CUDA.@sync cDFT.δFδρ_res_newautodiff!(
            $system_gpu,
            $ρ0_gpu,
            $δfδρ_new_gpu,
            $(cache_new_gpu)...,
        )

        new_gpu_host_mem = BenchmarkTools.memory(bench_new_gpu)
        new_gpu_host_allocs = BenchmarkTools.allocs(bench_new_gpu)

        CUDA.synchronize()

        # ============================================================
        # Correctness comparison
        # ============================================================
        old_gpu_arr = Array(δfδρ_old_gpu)
        new_gpu_arr = Array(δfδρ_new_gpu)

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

        @printf(
            "%8d  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %14s  %10d  %10d  %10d  %10d  %12.3f  %12.3f  %12.3f  %12.4e  %12.4e\n",
            NGRID,
            old_cpu_min_ms,
            old_cpu_med_ms,
            old_gpu_min_ms,
            old_gpu_med_ms,
            new_cpu_min_ms,
            new_cpu_med_ms,
            new_gpu_min_ms,
            new_gpu_med_ms,
            bytes_human(old_cpu_mem),
            bytes_human(new_cpu_mem),
            bytes_human(old_gpu_host_mem),
            bytes_human(new_gpu_host_mem),
            bytes_human(old_gpu_prealloc_mem),
            bytes_human(new_gpu_prealloc_mem),
            bytes_human(old_gpu_run_mem),
            bytes_human(new_gpu_run_mem),
            old_cpu_allocs,
            new_cpu_allocs,
            old_gpu_host_allocs,
            new_gpu_host_allocs,
            speedup_cpu,
            speedup_gpu,
            speedup_new_gpu_cpu,
            diff_new_cpu,
            diff_new_gpu,
        )

        # ============================================================
        # Free GPU-side objects before next NGRID
        # ============================================================
        old_gpu_arr = nothing
        new_gpu_arr = nothing
        cache_old_gpu = nothing
        cache_new_gpu = nothing
        δfδρ_old_gpu = nothing
        δfδρ_new_gpu = nothing
        old_prealloc_result = nothing
        new_prealloc_result = nothing
        ρ0_gpu = nothing
        system_gpu = nothing
        structure_gpu = nothing

        GC.gc()
        CUDA.reclaim()
    end
end

main()