###############################################################################
# check_rock2_mem.jl
#
# Purpose:
#   Diagnose whether ROCK2 / solver internals cause GPU memory pool growth.
#
# Required before include():
#   system
#   ρ0
#
# Usage:
#   ρ0 = ρ   # if your initial density is named ρ
#   include("benchmark/check_rock2_mem.jl")
###############################################################################

using CUDA
using KernelAbstractions
using SciMLBase
using DifferentialEquations
using DiffEqCallbacks
using cDFT

# OrdinaryDiffEq is often where Tsit5/ROCK2 live.
# This import may fail in some environments, so keep it non-fatal.
try
    @eval import OrdinaryDiffEq
catch err
    @warn "Could not import OrdinaryDiffEq. Will try DifferentialEquations/Main for algorithms." exception=(err, catch_backtrace())
end

###############################################################################
# Safety checks
###############################################################################

if !(@isdefined system)
    error("`system` is not defined. Define it before include(\"benchmark/check_rock2_mem.jl\").")
end

if !(@isdefined ρ0)
    if @isdefined ρ
        @warn "`ρ0` is not defined, but `ρ` exists. Using `ρ` as ρ0."
        ρ0 = ρ
    else
        error("Neither `ρ0` nor `ρ` is defined. Define initial density as `ρ0`.")
    end
end

###############################################################################
# Algorithm resolver
###############################################################################

function _module_has(m::Module, s::Symbol)
    try
        return isdefined(m, s)
    catch
        return false
    end
end

function get_alg(name::Symbol)
    # Try Main first, then OrdinaryDiffEq if available, then DifferentialEquations.
    mods = Module[Main]

    if @isdefined OrdinaryDiffEq
        push!(mods, OrdinaryDiffEq)
    end

    push!(mods, DifferentialEquations)

    for m in mods
        if _module_has(m, name)
            alg_ctor = getproperty(m, name)
            return alg_ctor()
        end
    end

    error("Could not find algorithm `$name`. Available modules tried: Main, OrdinaryDiffEq if imported, DifferentialEquations.")
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

function safe_total_memory()
    try
        return CUDA.total_memory()
    catch
        return nothing
    end
end

function safe_available_memory()
    try
        return CUDA.available_memory()
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

    total = safe_total_memory()
    avail = safe_available_memory()

    println()
    println("CUDA available memory = ", fmt_bytes(avail))
    println("CUDA total memory     = ", fmt_bytes(total))

    if total !== nothing && avail !== nothing
        println("CUDA used memory      = ", fmt_bytes(total - avail))
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
# Optional callback:
#   Print memory during solve every N callback calls.
###############################################################################

function make_mem_print_callback(label; every=20, do_reclaim=false)
    counter = Ref(0)

    condition(u, t, integrator) = true

    function affect!(integrator)
        counter[] += 1

        if counter[] % every == 0
            if do_reclaim
                gpu_reclaim("$label callback at count $(counter[]), t=$(integrator.t)")
            else
                gpu_mem("$label callback at count $(counter[]), t=$(integrator.t)")
            end
        end

        return nothing
    end

    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

###############################################################################
# One solver run
###############################################################################

function run_solver_case(
    case_name,
    alg,
    system,
    ρ_init;
    tspan=(0.0, 1.0),
    dt=nothing,
    dtmax=nothing,
    adaptive=nothing,
    callback=nothing,
    progress=false,
)
    println()
    println("============================================================")
    println("Running case: ", case_name)
    println("Algorithm: ", alg)
    println("tspan: ", tspan)
    println("dt: ", dt)
    println("dtmax: ", dtmax)
    println("adaptive: ", adaptive)
    println("============================================================")

    ρ = copy(ρ_init)
    prob = SciMLBase.ODEProblem(system, ρ, tspan)

    gpu_reclaim("$case_name: before solve, after reclaim")

    kwargs = Dict{Symbol, Any}()

    kwargs[:save_everystep] = false
    kwargs[:save_start] = false
    kwargs[:save_end] = false
    kwargs[:dense] = false
    kwargs[:saveat] = Float64[]
    kwargs[:progress] = progress
    kwargs[:progress_steps] = 1

    if dt !== nothing
        kwargs[:dt] = dt
    end

    if dtmax !== nothing
        kwargs[:dtmax] = dtmax
    end

    if adaptive !== nothing
        kwargs[:adaptive] = adaptive
    end

    if callback !== nothing
        kwargs[:callback] = callback
    end

    sol = nothing
    err = nothing

    elapsed = @elapsed begin
        try
            sol = solve(prob, alg; kwargs...)
            gpu_sync()
        catch e
            err = e
        end
    end

    println()
    println("Case finished: ", case_name)
    println("Elapsed seconds: ", elapsed)

    if err !== nothing
        println("ERROR in case: ", case_name)
        showerror(stdout, err)
        println()
    else
        println("retcode: ", sol.retcode)
        println("length(sol.t): ", length(sol.t))
        println("length(sol.u): ", length(sol.u))
    end

    gpu_mem("$case_name: after solve, before reclaim")
    gpu_reclaim("$case_name: after solve, after reclaim")

    # Drop references before next case.
    sol = nothing
    prob = nothing
    ρ = nothing
    err = nothing

    GC.gc(true)
    gpu_reclaim("$case_name: after dropping references and reclaim")

    return nothing
end

###############################################################################
# Main diagnostic suite
###############################################################################

function check_rock2_memory(
    system,
    ρ_init;
    tspan=(0.0, 1.0),
    baseline_dt=0.001,
    rock2_fixed_dt=0.001,
    rock2_dtmax=0.05,
    callback_every=20,
)
    println()
    println("============================================================")
    println("ROCK2 GPU memory diagnostic started")
    println("============================================================")
    println("system type = ", typeof(system))
    println("backend     = ", system.options.device)
    println("ngrid       = ", system.structure.ngrid)
    println("ρ size      = ", size(ρ_init))
    println("ρ type      = ", typeof(ρ_init))
    println("tspan       = ", tspan)
    println("baseline_dt = ", baseline_dt)
    println("rock2_fixed_dt = ", rock2_fixed_dt)
    println("rock2_dtmax = ", rock2_dtmax)
    println("============================================================")

    gpu_reclaim("Initial state")

    # Resolve algorithms once.
    baseline_alg = get_alg(:Tsit5)
    rock2_alg_1 = get_alg(:ROCK2)
    rock2_alg_2 = get_alg(:ROCK2)
    rock2_alg_3 = get_alg(:ROCK2)
    rock2_alg_4 = get_alg(:ROCK2)
    rock2_alg_5 = get_alg(:ROCK2)

    # -----------------------------------------------------------------------
    # Case 1: Tsit5 fixed-step baseline
    #
    # This is not the same stability class as ROCK2, but it checks whether
    # generic solve() with a GPU CuArray state causes the same pool growth.
    # -----------------------------------------------------------------------
    run_solver_case(
        "Case 1: Tsit5 fixed-step baseline",
        baseline_alg,
        system,
        ρ_init;
        tspan=tspan,
        dt=baseline_dt,
        adaptive=false,
        progress=false,
    )

    # -----------------------------------------------------------------------
    # Case 2: ROCK2 default adaptive with dtmax
    # This is close to your current usage.
    # -----------------------------------------------------------------------
    run_solver_case(
        "Case 2: ROCK2 adaptive dtmax",
        rock2_alg_1,
        system,
        ρ_init;
        tspan=tspan,
        dtmax=rock2_dtmax,
        progress=true,
    )

    # -----------------------------------------------------------------------
    # Case 3: ROCK2 fixed step
    # If fixed-step ROCK2 is much better, adaptive/controller/rejected steps
    # are likely involved.
    # -----------------------------------------------------------------------
    run_solver_case(
        "Case 3: ROCK2 fixed-step adaptive=false",
        rock2_alg_2,
        system,
        ρ_init;
        tspan=tspan,
        dt=rock2_fixed_dt,
        adaptive=false,
        progress=true,
    )

    # -----------------------------------------------------------------------
    # Case 4: ROCK2 adaptive with smaller dtmax
    # If smaller dtmax changes memory behavior, stage count / step logic matters.
    # -----------------------------------------------------------------------
    run_solver_case(
        "Case 4: ROCK2 adaptive smaller dtmax",
        rock2_alg_3,
        system,
        ρ_init;
        tspan=tspan,
        dtmax=rock2_dtmax / 10,
        progress=true,
    )

    # -----------------------------------------------------------------------
    # Case 5: ROCK2 adaptive with memory-print callback.
    # This prints during solve, without reclaim.
    # -----------------------------------------------------------------------
    cb_print = make_mem_print_callback(
        "Case 5: ROCK2 memory print callback";
        every=callback_every,
        do_reclaim=false,
    )

    run_solver_case(
        "Case 5: ROCK2 adaptive with memory print callback",
        rock2_alg_4,
        system,
        ρ_init;
        tspan=tspan,
        dtmax=rock2_dtmax,
        callback=cb_print,
        progress=true,
    )

    # -----------------------------------------------------------------------
    # Case 6: ROCK2 adaptive with periodic CUDA.reclaim callback.
    # If this stays bounded, the large usage is reclaimable pool/cache.
    # -----------------------------------------------------------------------
    cb_reclaim = make_mem_print_callback(
        "Case 6: ROCK2 reclaim callback";
        every=callback_every,
        do_reclaim=true,
    )

    run_solver_case(
        "Case 6: ROCK2 adaptive with periodic CUDA.reclaim",
        rock2_alg_5,
        system,
        ρ_init;
        tspan=tspan,
        dtmax=rock2_dtmax,
        callback=cb_reclaim,
        progress=true,
    )

    println()
    println("============================================================")
    println("ROCK2 GPU memory diagnostic finished")
    println("============================================================")

    return nothing
end

###############################################################################
# Run
###############################################################################

check_rock2_memory(
    system,
    ρ0;
    tspan=(0.0, 1.0),
    baseline_dt=0.001,
    rock2_fixed_dt=0.001,
    rock2_dtmax=0.05,
    callback_every=20,
)