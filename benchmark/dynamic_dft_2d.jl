import Logging: global_logger
import TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using Pkg, CUDA, HDF5, OrdinaryDiffEqStabilizedRK, DiffEqCallbacks
using DifferentialEquations
# using Revise
# Pkg.activate("..")
using Clapeyron, cDFT

# Define model
model = PCSAFT(["hexane"])

# Simulation conditions
T = 298.15
z = [1.0]
p, vl, vv = saturation_pressure(model, T)

ρ1 = z ./ vl
ρ2 = z ./ vv

ρb = (ρ1+ρ2*4)/5

# Define DFT structure
L = cDFT.length_scale(model) # Useful length scale for non-dimensionalization and grid size choice

ngrid = 101
println("Using grid size: ", ngrid, "x", ngrid)

structure = cDFT.Uniform2DCart((p, T), ρb, [-10L 10L; -10L 10L], (ngrid, ngrid));

options = DFTOptions(CUDABackend())
# options = DFTOptions(CPU())
println("Using device: ", options.device)

system = DFTSystem(model, structure, options)

ρ0 = cDFT.initialize_profiles(system);

δ = 0.01
for α in 1:1
    ρ0[:,:, α] .+= ρb[α] .* δ .*(-1)^α .* CUDA.rand(ngrid, ngrid, 1)
end

h5path = "trajectory_test_2d.h5"
rm(h5path; force=true)

h5open(h5path, "w") do f
    f["created"] = 1
end

step_counter = Ref(0)
save_every   = 1

saving_callback = FunctionCallingCallback(func_everystep=true) do ρ, t, integrator
    step_counter[] += 1

    if step_counter[] % save_every == 0
        CUDA.synchronize()
        rho_cpu = Array(exp.(ρ))

        h5open(h5path, "r+") do f
            f["rho_$(step_counter[])"] = rho_cpu
            f["t_$(step_counter[])"] = t
        end
    end
end

prob = ODEProblem(system, ρ0, (0.0, 250.0))

sol = solve(
    prob,
    ROCK2();
    callback=saving_callback,
    dtmax=0.05,
    save_everystep=false,
    saveat=[],
    progress=true,
    progress_steps=1,
)