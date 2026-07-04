# Figures for docs/src/tutorials/curved_surface_adsorption.md
include("common.jl")
using Clapeyron, cDFT, CairoMakie

model = PCSAFT(["ethane"])
T, p = 298.0, 1e7
v = Clapeyron.volume(model, p, T, [1.0]; phase=:liquid)
ρbulk = [1/v]
L = cDFT.length_scale(model)
width = 10L
surface = Steele(["graphite"], width)

# Inside a cylindrical pore, radius 10L.
structure_in = Uniform1DCyl((p, T), ρbulk, [0.0, width-0.5L], 201)
system_in = DFTSystem(model, structure_in, surface)
ρ_in = cDFT.initialize_profiles(system_in)
converge!(system_in, ρ_in)
save(assetpath("curved_surface_adsorption_cyl_inside.png"), plot(system_in, ρ_in))

println("saved curved_surface_adsorption_{cyl_inside,cyl_outside,sphr}.png to ", ASSETS)
