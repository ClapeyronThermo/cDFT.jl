###############################################################################
# mem_diagnosis.jl
#
# Purpose:
#   Diagnose GPU memory accumulation for:
#   1. δFδρ_res_newautodiff! alone
#   2. full δFδρ_res_newautodiff
#   3. ODE RHS directly, without solve()
#   4. solve() with all saving disabled
#
# Usage:
#   1. Put your system construction in USER SETUP SECTION.
#   2. Make sure `system` and `ρ0` are defined.
#   3. Run:
#        include("benchmark/mem_diagnosis.jl")
#
###############################################################################

using CUDA
using KernelAbstractions
using SciMLBase
using DifferentialEquations
using cDFT

###############################################################################
# USER SETUP SECTION
#
# Replace this section with your actual setup.
#
# Required after this section:
#   system
#   ρ0
#
# IMPORTANT:
#   This section should only build the model/system/initial density.
#   Do NOT call solve() here.
###############################################################################

# ---------------------------------------------------------------------------
# Option A: include your existing setup file
# ---------------------------------------------------------------------------
# include("your_setup_file.jl")

# ---------------------------------------------------------------------------
# Option B: if you already define system and ρ0 in REPL before include(),
# leave this section empty.
# ---------------------------------------------------------------------------

###############################################################################
# Safety checks
###############################################################################

if !(@isdefined system)
    error("`system` is not defined. Define it in USER SETUP SECTION or before include().")
end

if !(@isdefined ρ0)
    if @isdefined ρ
        @warn "`ρ0` is not defined, but `ρ` is defined. Using `ρ` as initial density."
        ρ0 = ρ
    else
        error("Neither `ρ0` nor `ρ` is defined. Define initial density as `ρ0`.")
    end
end

###############################################################################
# CUDA memory helpers
###############################################################################

function fmt_bytes(x)
    x === nothing && return "unknown"

    y = Float64(x)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    i = 1

    while y >= 1024 && i < length(units)
        y /= 1024
        i += 1
    end

    return string(round(y; digits=3), " ", units[i])
end

function safe_cuda_available_memory()
    try
        return CUDA.available_memory()
    catch
        return nothing
    end
end

function safe_cuda_total_memory()
    try
        return CUDA.total_memory()
    catch
        return nothing
    end
end

function print_cuda_memory()
    println("CUDA memory report:")

    if isdefined(CUDA, :pool_status)
        try
            println()
            println("CUDA pool_status():")
            CUDA.pool_status()
        catch err
            println("CUDA.pool_status() exists but failed:")
            println(err)
        end
    else
        println("CUDA.pool_status() is not available in this CUDA.jl version.")
    end

    free_mem = safe_cuda_available_memory()
    total_mem = safe_cuda_total_memory()

    println()
    println("CUDA available memory = ", fmt_bytes(free_mem))
    println("CUDA total memory     = ", fmt_bytes(total_mem))

    if free_mem !== nothing && total_mem !== nothing
        used_mem = total_mem - free_mem
        println("CUDA used memory      = ", fmt_bytes(used_mem))
    end

    return nothing
end

function gpu_sync()
    try
        CUDA.synchronize()
    catch err
        println("CUDA.synchronize() failed:")
        println(err)
    end

    return nothing
end

function gpu_mem(tag; do_gc=true)
    gpu_sync()

    if do_gc
        GC.gc(false)
    end

    println()
    println("================================================================")
    println(tag)
    println("================================================================")
    print_cuda_memory()
    println("================================================================")
    println()

    return nothing
end

function gpu_reclaim(tag)
    gpu_sync()
    GC.gc(true)

    try
        CUDA.reclaim()
    catch err
        println("CUDA.reclaim() failed:")
        println(err)
    end

    gpu_sync()

    println()
    println("################################################################")
    println(tag)
    println("################################################################")
    print_cuda_memory()
    println("################################################################")
    println()

    return nothing
end

###############################################################################
# Test 1:
# Repeat only δFδρ_res_newautodiff!
#
# This isolates:
#   evaluate_field!
#   copyto!(n, fft_buf)
#   fill!(δf_val, 1.0)
#   fill!(δf, 0.0)
#   δf_enzyme_kernel!
#   copyto!(fft_buf, δf)
#   integrate_field!
#
# It does NOT include:
#   evaluate_external_field!
#   propagate!
#   ODE RHS FFT/convolve
#   solve()
###############################################################################

function test_newautodiff_bang_repeat(system, ρ_init; ntest=100, every=10)
    println()
    println("###################################")
    println("# Test 1: δFδρ_res_newautodiff!  #")
    println("###################################")

    ρ = copy(ρ_init)

    δfδρ_res, cache_model, cache_external, cache_propagator =
        cDFT.preallocate_newautodiff(system, ρ)

    gpu_reclaim("Test 1 start: after reclaim")

    for i in 1:ntest
        cDFT.δFδρ_res_newautodiff!(
            system,
            ρ,
            δfδρ_res,
            cache_model...
        )

        gpu_sync()

        if i == 1 || i % every == 0 || i == ntest
            gpu_mem("Test 1: after δFδρ_res_newautodiff! iter $i")
        end
    end

    gpu_mem("Test 1 end: before reclaim")
    gpu_reclaim("Test 1 end: after reclaim")

    return nothing
end

###############################################################################
# Test 1b:
# Repeat full δFδρ_res_newautodiff(system, ρ)
#
# This includes allocation inside the non-mutating API each call:
#   preallocate_newautodiff
#   δFδρ_res_newautodiff!
#   evaluate_external_field!
#   propagate!
#
# If this grows but Test 1 does not, the issue may simply be allocations from
# using the non-mutating API repeatedly.
###############################################################################

function test_newautodiff_full_repeat(system, ρ_init; ntest=100, every=10)
    println()
    println("######################################")
    println("# Test 1b: δFδρ_res_newautodiff API #")
    println("######################################")

    ρ = copy(ρ_init)

    gpu_reclaim("Test 1b start: after reclaim")

    for i in 1:ntest
        μ = cDFT.δFδρ_res_newautodiff(system, ρ)
        gpu_sync()

        # Explicitly drop reference.
        μ = nothing

        if i == 1 || i % every == 0 || i == ntest
            GC.gc(false)
            gpu_mem("Test 1b: after δFδρ_res_newautodiff iter $i")
        end
    end

    gpu_mem("Test 1b end: before reclaim")
    gpu_reclaim("Test 1b end: after reclaim")

    return nothing
end

###############################################################################
# Test 2:
# Repeat ODE RHS directly, without solve()
#
# This tests the closure created by:
#   SciMLBase.ODEProblem(system, ρ, tspan)
#
# If Test 1 is stable but Test 2 grows, the issue is in RHS composition:
#   newautodiff + external + propagate + convolve + broadcasts
###############################################################################

function test_rhs_repeat(system, ρ_init; ntest=100, every=10)
    println()
    println("#####################################")
    println("# Test 2: direct RHS, without solve #")
    println("#####################################")

    ρ = copy(ρ_init)

    prob = SciMLBase.ODEProblem(system, ρ, (0.0, 1.0))

    η = copy(prob.u0)
    dη = similar(η)
    fill!(dη, 0.0)

    gpu_reclaim("Test 2 start: after reclaim")

    for i in 1:ntest
        prob.f(dη, η, prob.p, 0.0)
        gpu_sync()

        if i == 1 || i % every == 0 || i == ntest
            gpu_mem("Test 2: after RHS iter $i")
        end
    end

    gpu_mem("Test 2 end: before reclaim")
    gpu_reclaim("Test 2 end: after reclaim")

    return nothing
end

###############################################################################
# Test 3:
# solve() with saving disabled as much as possible.
#
# If Test 2 is stable but Test 3 grows, the issue is likely solver-side:
#   saving
#   dense interpolation
#   step cache
#   rejected steps
#   internal GPU state retention
###############################################################################

function test_solve_minimal(system, ρ_init; tspan=(0.0, 1.0), dtmax=0.05)
    println()
    println("####################################")
    println("# Test 3: solve with minimal saves #")
    println("####################################")

    ρ = copy(ρ_init)

    prob = SciMLBase.ODEProblem(system, ρ, tspan)

    gpu_reclaim("Test 3 start: after reclaim")

    sol = solve(
        prob,
        ROCK2();
        dtmax=dtmax,
        save_everystep=false,
        save_start=false,
        save_end=false,
        dense=false,
        saveat=Float64[],
        progress=true,
        progress_steps=1,
    )

    gpu_sync()

    gpu_mem("Test 3 end: after solve, before reclaim")
    gpu_reclaim("Test 3 end: after reclaim")

    return sol
end

###############################################################################
# Test 4:
# solve() with periodic CUDA.reclaim() callback.
#
# This is not a real fix. It tests whether CUDA.reclaim() keeps memory bounded.
#
# If memory is controlled by this callback, accumulation is likely CUDA pool/cache
# behavior or temporary GPU allocation churn.
###############################################################################

function test_solve_with_reclaim_callback(
    system,
    ρ_init;
    tspan=(0.0, 1.0),
    dtmax=0.05,
    reclaim_every=20,
)
    println()
    println("############################################")
    println("# Test 4: solve with periodic CUDA.reclaim #")
    println("############################################")

    ρ = copy(ρ_init)

    prob = SciMLBase.ODEProblem(system, ρ, tspan)

    counter = Ref(0)

    condition(u, t, integrator) = true

    function affect!(integrator)
        counter[] += 1

        if counter[] % reclaim_every == 0
            println()
            println("Callback reclaim at callback counter = ", counter[])
            gpu_reclaim("Test 4 callback reclaim")
        end

        return nothing
    end

    cb = DiscreteCallback(condition, affect!; save_positions=(false, false))

    gpu_reclaim("Test 4 start: after reclaim")

    sol = solve(
        prob,
        ROCK2();
        callback=cb,
        dtmax=dtmax,
        save_everystep=false,
        save_start=false,
        save_end=false,
        dense=false,
        saveat=Float64[],
        progress=true,
        progress_steps=1,
    )

    gpu_sync()

    gpu_mem("Test 4 end: after solve, before reclaim")
    gpu_reclaim("Test 4 end: after reclaim")

    return sol
end

###############################################################################
# Run selected tests
###############################################################################

function run_all_memory_tests(
    system,
    ρ_init;
    ntest=100,
    tspan=(0.0, 1.0),
    dtmax=0.05,
    run_test_1=true,
    run_test_1b=false,
    run_test_2=true,
    run_test_3=true,
    run_test_4=false,
)
    println()
    println("====================================================")
    println("GPU memory diagnosis started")
    println("====================================================")
    println("system type = ", typeof(system))
    println("backend     = ", system.options.device)
    println("ngrid       = ", system.structure.ngrid)
    println("ρ size      = ", size(ρ_init))
    println("ρ type      = ", typeof(ρ_init))
    println("ntest       = ", ntest)
    println("tspan       = ", tspan)
    println("dtmax       = ", dtmax)
    println("====================================================")

    gpu_reclaim("Initial state after CUDA.reclaim")

    if run_test_1
        test_newautodiff_bang_repeat(
            system,
            ρ_init;
            ntest=ntest,
            every=max(1, ntest ÷ 10),
        )
    end

    if run_test_1b
        test_newautodiff_full_repeat(
            system,
            ρ_init;
            ntest=ntest,
            every=max(1, ntest ÷ 10),
        )
    end

    if run_test_2
        test_rhs_repeat(
            system,
            ρ_init;
            ntest=ntest,
            every=max(1, ntest ÷ 10),
        )
    end

    sol = nothing

    if run_test_3
        sol = test_solve_minimal(
            system,
            ρ_init;
            tspan=tspan,
            dtmax=dtmax,
        )
    end

    if run_test_4
        sol = test_solve_with_reclaim_callback(
            system,
            ρ_init;
            tspan=tspan,
            dtmax=dtmax,
            reclaim_every=max(1, ntest ÷ 5),
        )
    end

    println()
    println("====================================================")
    println("GPU memory diagnosis finished")
    println("====================================================")

    return sol
end

###############################################################################
# Main
#
# Default:
#   Run Test 1, Test 2, Test 3.
#   Do not run Test 1b by default because it intentionally reallocates each call.
#   Do not run Test 4 by default because it is only a reclaim experiment.
###############################################################################

sol = run_all_memory_tests(
    system,
    ρ0;
    ntest=100,
    tspan=(0.0, 1.0),
    dtmax=0.05,
    run_test_1=true,
    run_test_1b=false,
    run_test_2=true,
    run_test_3=true,
    run_test_4=false,
)