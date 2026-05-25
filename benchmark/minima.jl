#!/usr/bin/env julia

using cDFT
using cDFT.Clapeyron
using Printf
using LinearAlgebra
using BenchmarkTools

# Adjustable grid count:
const NGRID = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128

# Physical state:
const T = 300.0
const p = 1.0e5

# Example component. Hexane is liquid at 300K, 1atm.
components = ["hexane"]
model = PCSAFT(components)

# Bulk density (Keep it strictly in mol/m³)
vl = volume(model, p, T)
ρbulk = [1.0 / vl] 

# Length scale
L = cDFT.length_scale(model)
println("Length scale L = $(L) m")

# 1D DFT structure/domain (Uniform for benchmarking against bulk)
structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], NGRID)
println("Created a structure with bulk density: $(ρbulk[1]) mol/m³")

# Build system
system = DFTSystem(model, structure)

# Initial density profile
ρ0 = cDFT.initialize_profiles(system)

# Evaluate the free energy and derivative
F = cDFT.F_res(system, ρ0)
dF = cDFT.δFδρ_res(system, ρ0)

# Functional derivative validation (Dimensionless: μ / RT)
μ_bulk = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T

max_err = maximum(abs.(dF .- μ_bulk[1]))
@printf("NGRID=%d F_res=%.8e\n", NGRID, F)
@printf("Bulk μ_res/RT (Clapeyron): %.8f\n", μ_bulk[1])
@printf("Max absolute error vs dF/dρ: %.8e\n", max_err)

# Performance Profiling
println("\nProfiling δFδρ_res performance:")
@btime cDFT.δFδρ_res($system, $ρ0)

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
