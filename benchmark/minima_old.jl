#!/usr/bin/env julia

using cDFT
using cDFT.Clapeyron
using Printf
using LinearAlgebra
using BenchmarkTools
using CUDA

# Adjustable grid count
NGRID = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4096

# Physical state
# -----------------------------------------------------------------------------
# Example component: hexane is liquid at 300 K, 1 atm
T = 300.0
p = 1.0e5
components = ["hexane"]
model = PCSAFT(components)
# Bulk density, strictly in mol/m^3
vl = volume(model, p, T)
ρbulk = [1.0 / vl]
# Length scale
L = cDFT.length_scale(model)
println("Length scale L = $(L) m")
# -----------------------------------------------------------------------------

# 1D DFT structure/domain: uniform bulk profile for benchmarking
structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], NGRID)
println("Created structure with bulk density: $(ρbulk[1]) mol/m^3")

options = DFTOptions(CUDABackend())
# options = DFTOptions(CPU())

# Build system
system = DFTSystem(model, structure, options)

# Initial density profile
ρ0 = cDFT.initialize_profiles(system)
println("typeof(ρ0) = ", typeof(ρ0))

# Optional: check where model buffers are allocated
δfδρ_tmp, cache_model_tmp, _, _ = cDFT.preallocate(system, ρ0)
n_tmp, δf_tmp, fft_buf_tmp, in_buf_tmp, out_buf_tmp, P_tmp, iP_tmp, f_tmp, cache_pool_tmp = cache_model_tmp

println("typeof(n)       = ", typeof(n_tmp))
println("typeof(δf)      = ", typeof(δf_tmp))
println("typeof(fft_buf) = ", typeof(fft_buf_tmp))
println("typeof(in_buf)  = ", typeof(in_buf_tmp))
println("typeof(out_buf) = ", typeof(out_buf_tmp))

# Do NOT call F_res here.
# F_res is not GPU-safe yet because it scalar-indexes a CuArray.
# F = cDFT.F_res(system, ρ0)

# Warm-up / correctness evaluations
dF_old = cDFT.δFδρ_res(system, ρ0)

# Reference bulk residual chemical potential, dimensionless μ_res / RT
μ_bulk = Clapeyron.VT_chemical_potential_res(
    model,
    1 / sum(ρbulk),
    T,
    ρbulk / sum(ρbulk),
) / Clapeyron.R̄ / T

# Correctness checks
err_old_vs_bulk = maximum(abs.(dF_old .- μ_bulk[1]))

@printf("NGRID=%d\n", NGRID)
@printf("Bulk μ_res/RT from Clapeyron: %.8f\n", μ_bulk[1])
@printf("Max abs error: old vs bulk              = %.8e\n", err_old_vs_bulk)

# Benchmarking
println("\nBenchmarking δFδρ_res:")
t = @benchmark cDFT.δFδρ_res!($system, $ρ0, $δfδρ_tmp, $cache_model_tmp...)
t_bench = median(t.times)./ 1e6
u_t_bench = std(t.times)./ 1e6
println(": time = ", t_bench, " ms; std = ", u_t_bench, " ms")

