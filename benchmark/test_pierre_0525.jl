using Pkg, BenchmarkTools, Revise, CUDA
using Clapeyron, cDFT, FFTW

# include("preallocate.jl")

model = PCSAFT(["hexane"])

T = 298.15
(p, vl, vv) = saturation_pressure(model, T);

ρl = [1.]/vl
ρv = [1.]/vv

L = cDFT.length_scale(model) # Useful tool to obtain a characteristic length scale for the system, which can be used to non-dimensionalize the problem and choose an appropriate grid size.

N = [101]

options = DFTOptions(CPU())

t_bench = zeros(length(N))
u_t_bench = zeros(length(N))

for (i, n) in enumerate(N)
    # if n != N[end]
    #     println("N = ", n, ": time = ", t_bench[i], " ns; std = ", u_t_bench[i], " ns")
    #     continue
    # end
    structure = Uniform1DCart((p, T), ρl, [-10L,10L], (n,));

    system = DFTSystem(model, structure, options)

    ρ = cDFT.initialize_profiles(system);

    δfδρ_res, cache_model, _ = cDFT.preallocate(system, ρ)

    cDFT.δFδρ_res!(system, ρ, δfδρ_res, cache_model...)

    t = @benchmark cDFT.δFδρ_res!($system, $ρ, $δfδρ_res, $cache_model...)
    t_bench[i] = median(t.times)./ 1e6
    u_t_bench[i] = std(t.times)./ 1e6
    println("N = ", n, ": time = ", t_bench[i], " μs; std = ", u_t_bench[i], " μs")
end

# df = DataFrame(N=N, time=t_bench, u_time=u_t_bench)

# name = "benchmark_uniform_"*string(Threads.nthreads())*"_threads"
# if options.device isa CUDABackend
#     name *= "_gpu"
# end
# CSV.write(name*".csv", df)

# cDFT.δFδρ_res(system, ρ, δfδρ_res,cache_model...)
# # cDFT.evaluate_field!(system, ρ, cache, in_buf, out_buf, plan, iplan)