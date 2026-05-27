import Logging: global_logger
import TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using Pkg, CUDA, HDF5, DifferentialEquations, DiffEqCallbacks, Revise
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

ngrid = 51

structure = cDFT.Uniform2DCart((p, T), ρb, [-10L 10L; -10L 10L], (ngrid, ngrid));

options = DFTOptions(CUDABackend())
# options = DFTOptions(CPU())

system = DFTSystem(model, structure, options)

ρ0 = cDFT.initialize_profiles(system);

δ = 0.01
for α in 1:1
    # ρ0[:,:, α] .+= ρb[α] .* δ .*(-1)^α .* rand(ngrid, ngrid, 1)
    ρ0[:,:, α] .+= ρb[α] .* δ .*(-1)^α .* CUDA.rand(ngrid, ngrid, 1)
end

h5file = h5open("trajectory_test_2d_vle.h5", "w")
step_counter = Ref(0)
save_every   = 1    # save every N steps

saving_callback = FunctionCallingCallback(func_everystep=true) do ρ, t, integrator
    step_counter[] += 1
    if step_counter[] % save_every == 0
        h5write("trajectory_test_2d_vle.h5", "rho_$(step_counter[])", Array(exp.(ρ)))
        h5write("trajectory_test_2d_vle.h5", "t_$(step_counter[])", t)
    end
end

prob = ODEProblem(system, ρ0, (0.0, 10000.0))

sol  = solve(prob, ROCK2();
callback=saving_callback,
    dtmax=0.05,
    save_everystep = false,
    saveat         = [],
    progress=true, 
    progress_steps=1)