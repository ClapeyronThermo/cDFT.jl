#!/usr/bin/env julia

using cDFT
using cDFT.Clapeyron
using Printf
using LinearAlgebra

# Adjustable grid count:
const NGRID = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128

# Physical state:
const T = 300.0
const p = 1.0e5

# Example component.
components = ["methane"]
model = PCSAFT(components)

# Bulk density
vl = volume(model, p, T)
ρbulk = [1.0/vl]

# Length scale
L = cDFT.length_scale(model)

# 1D DFT structure/domain (Uniform for benchmarking against bulk)
structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], NGRID)

# Build system
system = DFTSystem(model, structure)

# Initial density profile
ρ0 = cDFT.initialize_profiles(system)

# Evaluate the free energy and derivative
F = cDFT.F_res(system, ρ0)
dF = cDFT.δFδρ_res(system, ρ0)

@printf("NGRID=%d F_res=%.8e norm(dF)=%.8e\n", NGRID, F, norm(vec(dF)))

# Minimization API exists: converge!
# This will try to solve the equilibrium density profile.
# Since it is a uniform system, it should stay at ρbulk.
cDFT.converge!(system, ρ0)
@printf("Final density (first grid point): %.8f\n", ρ0[1])
