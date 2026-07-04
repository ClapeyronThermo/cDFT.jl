# Figure for docs/src/tutorials/gpu_acceleration.md
# Reuses the existing CPU/GPU benchmark CSVs already committed under examples/ (columns:
# N, time, u_time) rather than re-running the CUDA benchmark. If you'd rather regenerate
# them fresh, run examples/benchmark_uniform.jl and examples/benchmark_uniform_gpu.jl first
# (they overwrite the same CSV filenames used below).
include("common.jl")
using DelimitedFiles, CairoMakie

function read_bench(path)
    data, header = readdlm(path, ','; header=true)
    N, t, ut = data[:,1], data[:,2], data[:,3]
    return N, t, ut
end

cpu_path = joinpath(@__DIR__, "..", "benchmark_uniform_1_threads.csv")
gpu_path = joinpath(@__DIR__, "..", "benchmark_uniform_1_threads_gpu.csv")

N_cpu, t_cpu, u_cpu = read_bench(cpu_path)
N_gpu, t_gpu, u_gpu = read_bench(gpu_path)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="grid size N", ylabel="wall time / s", yscale=log10, xscale=log10)
lines!(ax, N_cpu, t_cpu; label="CPU (1 thread)", linewidth=3)
scatter!(ax, N_cpu, t_cpu)
lines!(ax, N_gpu, t_gpu; label="GPU", linewidth=3)
scatter!(ax, N_gpu, t_gpu)
axislegend(ax; position=:lt)
save(assetpath("gpu_acceleration_benchmark.png"), fig)

println("saved ", assetpath("gpu_acceleration_benchmark.png"))
