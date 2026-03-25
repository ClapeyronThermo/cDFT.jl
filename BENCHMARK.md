Develop scripts for benchmarking the speed and size scaling of the different backends in 1d, 2d, and 3d. These benchmarks should be professional as they will be used for publication in a computational journal.

Backends:
- CPU (Float64), with 4 threads (-t 4)
- CUDA (Float32 and Float64)
- Metal (Float32)

I will have to run the different backends on different machines, so they should be separate scripts. The goal is to get a sense of the performance for the different dimensionalities and how they scales with the number of gridpoints in each dimension. And then I will compare the results from the different backends.

Example of how to use CPU backend: examples/2d_CPU.jl
Example of how to use CUDA backend: examples/2d_CUDA.jl
Example of how to use Metal backend: examples/2d_METAL.jl

Default Polymer System:

N_seg    = 30
N_A      = 15
N_B      = N_seg - N_A
f_A      = N_A / N_seg
chi_val  = 1.0
nspecies = 2
rho0     = 1.0
kappa    = 20.0
b        = 1.0

Guidelines:
- Sample several system sizes for each dimensionality
- For 2d make a square, for 3d make a cube
- Do not full a full convergence, you only need to run a handful of steps (5-10) and average the time
- Do not include startup time, only include the actual computations that would normally be inside the SCFT loop
- Don't bother tracking convergence, we only care about the computation time itself
- Use BenchmarkTools from julia

Develop benchmark scripts for a polymeric system (diblock copolymer) and a separate benchmark for a monomeric (solvent) system. The main difference is that the monomeric system doesn't require propagators, and the density can be directly computed from the field.