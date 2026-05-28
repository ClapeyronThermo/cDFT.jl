import Logging: global_logger
import TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
using Pkg, CUDA, HDF5, DifferentialEquations, Revise
# Pkg.activate("..")
using Clapeyron, cDFT

# Define model
model = PCSAFT(["water","hexane"])

# Simulation conditions
p = 1e5
T = 290.15
z = [0.5, 0.5]

# Obtain composition of coexisting phases at given conditions
x, n, G = tp_flash(model, p, T, z, MichelsenTPFlash(equilibrium=:lle, K0 = [100., 0.001]))

# Obtain molar volumes of coexisting phases
v1 = volume(model, p, T, x[1,:])
v2 = volume(model, p, T, x[2,:])

# Obtain bulk densities of coexisting phases
ρ1 = x[1,:] ./ v1
ρ2 = x[2,:] ./ v2

ρb = (ρ1+ρ2)/2

# Define DFT structure
L = cDFT.length_scale(model) # Useful length scale for non-dimensionalization and grid size choice

ngrid = 51

structure = cDFT.Uniform3DCart((p, T), ρb, [-10L 10L; -10L 10L; -10L 10L], (ngrid, ngrid, ngrid));

options = DFTOptions(CUDABackend())

system = DFTSystem(model, structure, options)

ρ0 = cDFT.initialize_profiles(system);

δ = 0.01
for α in 1:2
    ρ0[:,:,:, α] .+= ρb[α] .* δ .*(-1)^α .* rand(ngrid, ngrid, ngrid, 1)
end

h5file = h5open("trajectory_test.h5", "w")
step_counter = Ref(0)
save_every   = 1    # save every N steps

saving_callback = FunctionCallingCallback(func_everystep=true) do ρ, t, integrator
    step_counter[] += 1
    if step_counter[] % save_every == 0
        h5write("trajectory_test.h5", "rho_$(step_counter[])", Array(exp.(ρ)))
        h5write("trajectory_test.h5", "t_$(step_counter[])", t)
    end
end

prob = ODEProblem(system, ρ0, (0.0, 1000.0))


sol  = solve(prob, ROCK2();callback=saving_callback,
    save_everystep = false,
    saveat         = [],
    progress=true, 
    progress_steps=1)