#!/usr/bin/env julia

using cDFT
using cDFT.Clapeyron
using Printf
using LinearAlgebra
using BenchmarkTools

# Adjustable grid count
const NGRID = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128

# Physical state
const T = 300.0
const p = 1.0e5

# Example component: hexane is liquid at 300 K, 1 atm
components = ["hexane"]
model = PCSAFT(components)

# Bulk density, strictly in mol/m^3
vl = volume(model, p, T)
ρbulk = [1.0 / vl]

# Length scale
L = cDFT.length_scale(model)
println("Length scale L = $(L) m")

# 1D DFT structure/domain: uniform bulk profile for benchmarking
structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], NGRID)
println("Created structure with bulk density: $(ρbulk[1]) mol/m^3")

# Build system
system = DFTSystem(model, structure)

# Initial density profile
ρ0 = cDFT.initialize_profiles(system)

# Warm-up / correctness evaluations
F = cDFT.F_res(system, ρ0)

dF_old = cDFT.δFδρ_res(system, ρ0)
dF_new = cDFT.δFδρ_res_newautodiff(system, ρ0)  # new autodiff version

# Reference bulk residual chemical potential, dimensionless μ_res / RT
μ_bulk = Clapeyron.VT_chemical_potential_res(
    model,
    1 / sum(ρbulk),
    T,
    ρbulk / sum(ρbulk),
) / Clapeyron.R̄ / T

# Correctness checks
err_old_vs_bulk = maximum(abs.(dF_old .- μ_bulk[1]))
err_new_vs_bulk = maximum(abs.(dF_new .- μ_bulk[1]))
err_new_vs_old  = maximum(abs.(dF_new .- dF_old))

@printf("NGRID=%d F_res=%.8e\n", NGRID, F)
@printf("Bulk μ_res/RT from Clapeyron: %.8f\n", μ_bulk[1])
@printf("Max abs error: old vs bulk      = %.8e\n", err_old_vs_bulk)
@printf("Max abs error: newautodiff vs bulk = %.8e\n", err_new_vs_bulk)
@printf("Max abs difference: newautodiff vs old = %.8e\n", err_new_vs_old)

# Benchmarking
println("\nBenchmarking δFδρ_res:")
bench_old = @benchmark cDFT.δFδρ_res($system, $ρ0)
display(bench_old)

println("\nBenchmarking δFδρ_res_newautodiff:")
bench_new = @benchmark cDFT.δFδρ_res_newautodiff($system, $ρ0)
display(bench_new)

# Simple timing summary
t_old = minimum(bench_old).time
t_new = minimum(bench_new).time
speedup = t_old / t_new

@printf("\nMinimum time old: %.3f μs\n", t_old / 1e3)
@printf("Minimum time newautodiff: %.3f μs\n", t_new / 1e3)
@printf("Speedup old / newautodiff: %.3fx\n", speedup)

# #!/usr/bin/env julia

# using cDFT
# using cDFT.Clapeyron
# using Printf
# using LinearAlgebra
# using BenchmarkTools

# # Adjustable grid count:
# const NGRID = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128

# # Physical state:
# const T = 300.0
# const p = 1.0e5

# # Example component. Hexane is liquid at 300K, 1atm.
# components = ["hexane"]
# model = PCSAFT(components)

# # Bulk density
# vl = volume(model, p, T)
# ρbulk = [1.0/vl] # Resulting unit: mol/m³

# # Length scale
# L = cDFT.length_scale(model)
# println("Length scale L = $(L) m")

# # 1D DFT structure/domain (Uniform for benchmarking against bulk)
# structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], NGRID)
# println("Createda structure with pressure: $(p) Pa, temperature: $(T) K, bulk density: $(ρbulk) mol/m³, domain: [-10L, 10L] m, NGRID: $(NGRID)")

# # Build system
# system = DFTSystem(model, structure)

# # Initial density profile
# ρ0 = cDFT.initialize_profiles(system)
# println("Initialized density profile ρ0 with length: ", length(ρ0))
# println("Initial density profile ρ0: ", ρ0)

# # Evaluate the free energy and derivative
# # for a uniform system, an easy check is: Clapeyron.VT_chemical_potential_res(model, v, T, [1.])
# # That should match dFres/drho
# F = cDFT.F_res(system, ρ0)

# # Functional derivative validation
# dF = cDFT.δFδρ_res(system, ρ0)
# μ_bulk = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T

# max_err = maximum(abs.(dF .- μ_bulk[1]))
# @printf("NGRID=%d F_res=%.8e\n", NGRID, F)
# @printf("Bulk μ_res (Clapeyron): %.8f\n", μ_bulk[1])
# @printf("Max absolute error vs dF/dρ: %.8e\n", max_err)

# # Performance Profiling
# println("\nProfiling δFδρ_res performance:")
# @btime cDFT.δFδρ_res($system, $ρ0)
