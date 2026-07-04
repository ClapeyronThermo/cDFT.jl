# Figure for docs/src/tutorials/getting_started.md
include("common.jl")
using Clapeyron, cDFT, CairoMakie

model = PCSAFT(["methane"])
T, p = 150.0, 1e7
v = Clapeyron.volume(model, p, T, [1.0]; phase=:liquid)
ρbulk = [1/v]
L = cDFT.length_scale(model)
width = 5L

surface = Steele(["graphite"], width)
structure = Uniform1DCart((p, T), ρbulk, [0.5L, width-0.5L], 201)
system = DFTSystem(model, structure, surface)

ρ = cDFT.initialize_profiles(system)
converge!(system, ρ)

fig = plot(system, ρ)
save(assetpath("getting_started_profile.png"), fig)
println("saved ", assetpath("getting_started_profile.png"))
