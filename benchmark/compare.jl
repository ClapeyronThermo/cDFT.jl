#!/usr/bin/env julia

using cDFT
using cDFT.Clapeyron
using Printf
using BenchmarkTools
using CUDA
using Statistics
using Plots

function run_benchmark(NGRID, use_gpu)
    T = 300.0
    p = 1.0e5
    components = ["hexane"]
    model = PCSAFT(components)

    vl = volume(model, p, T)
    ρbulk = [1.0 / vl]
    L = cDFT.length_scale(model)

    backend = use_gpu ? CUDABackend() : CPU()
    structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], NGRID)
    options = DFTOptions(backend)
    system = DFTSystem(model, structure, options)

    ρ0 = cDFT.initialize_profiles(system)
    δfδρ_res, cache, _, _ = cDFT.preallocate(system, ρ0)

    # Warmup
    if use_gpu
        CUDA.@sync cDFT.δFδρ_res!(system, ρ0, δfδρ_res, cache...)
    else
        cDFT.δFδρ_res!(system, ρ0, δfδρ_res, cache...)
    end

    # Timing
    n_samples = NGRID > 2^15 ? 3 : 10
    times = Float64[]
    for _ in 1:n_samples
        GC.gc()
        if use_gpu
            CUDA.reclaim()
            t = @elapsed CUDA.@sync cDFT.δFδρ_res!(system, ρ0, δfδρ_res, cache...)
        else
            t = @elapsed cDFT.δFδρ_res!(system, ρ0, δfδρ_res, cache...)
        end
        push!(times, t)
    end
    return median(times)
end

if length(ARGS) > 0 && ARGS[1] == "worker"
    mode = ARGS[2]
    ngrids = parse.(Int, split(ARGS[3], ","))
    use_gpu = (mode == "gpu")
    
    for NGRID in ngrids
        t = run_benchmark(NGRID, use_gpu)
        println("DATA:$NGRID:$t")
    end
    exit(0)
end

# Main Driver
NGRIDS = 2 .^ [5, 10, 15, 20, 25]
results = Dict()

modes = [
    ("1cpu", 1, false),
    ("4cpu", 4, false),
    ("gpu", 4, true)
]

for (name, threads, use_gpu) in modes
    println("Running $name benchmark...")
    cmd = `julia --project=.. compare.jl worker $name $(join(NGRIDS, ","))`
    env = copy(ENV)
    env["JULIA_NUM_THREADS"] = string(threads)
    
    output = read(setenv(cmd, env), String)
    
    data = []
    for line in split(output, "\n")
        if startswith(line, "DATA:")
            parts = split(line, ":")
            push!(data, (parse(Int, parts[2]), parse(Float64, parts[3])))
        end
    end
    results[name] = data
end

# Plotting
p = plot(xaxis=:log10, yaxis=:log10, xlabel="NGRID", ylabel="Time (s)", title="cDFT.δFδρ_res! Performance")

for (name, _) in modes
    data = results[name]
    x = [d[1] for d in data]
    y = [d[2] for d in data]
    plot!(p, x, y, label=name, marker=:circle)
end

savefig("compare.png")
println("Results saved to compare.png")

# Also print a summary table
@printf("\n%8s  %12s  %12s  %12s\n", "NGRID", "1cpu (s)", "4cpu (s)", "gpu (s)")
println("-"^50)
for i in 1:length(NGRIDS)
    @printf("%8d  %12.6f  %12.6f  %12.6f\n", 
        NGRIDS[i], 
        results["1cpu"][i][2], 
        results["4cpu"][i][2], 
        results["gpu"][i][2]
    )
end
