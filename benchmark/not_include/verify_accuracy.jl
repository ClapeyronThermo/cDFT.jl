using cDFT, Clapeyron, CUDA, Adapt, KernelAbstractions, LinearAlgebra, Printf

function verify_accuracy()
    if !CUDA.functional(); println("CUDA not functional"); return; end
    
    # Setup a simple 1D hexane system
    T = 300.0
    p = 1.0e5
    components = ["hexane"]
    model = PCSAFT(components)
    vl = volume(model, p, T)
    ρbulk = [1.0 / vl]
    L = cDFT.length_scale(model)
    
    ngrid = 128
    structure = Uniform1DCart((p, T), ρbulk, [-10L, 10L], ngrid)
    options = DFTOptions(CUDABackend())
    system = DFTSystem(model, structure, options)
    
    ρ0 = cDFT.initialize_profiles(system)
    ρ0_gpu = Adapt.adapt(CUDABackend(), ρ0)
    
    # Reference bulk residual chemical potential, dimensionless μ_res / RT
    μ_bulk = Clapeyron.VT_chemical_potential_res(
        model,
        1 / sum(ρbulk),
        T,
        ρbulk / sum(ρbulk),
    ) / Clapeyron.R̄ / T
    
    println("Bulk μ_res/RT from Clapeyron: ", μ_bulk[1])
    
    println("Calling δFδρ_res (Old)...")
    dF_old = cDFT.δFδρ_res(system, ρ0_gpu)
    
    println("Calling δFδρ_res_newautodiff (New)...")
    dF_new = cDFT.δFδρ_res_newautodiff(system, ρ0_gpu)
    
    err_old = maximum(abs.(Array(dF_old) .- μ_bulk[1]))
    err_new = maximum(abs.(Array(dF_new) .- μ_bulk[1]))
    diff = maximum(abs.(Array(dF_new) .- Array(dF_old)))
    
    @printf("Max abs error: old vs bulk              = %.8e\n", err_old)
    @printf("Max abs error: newautodiff vs bulk      = %.8e\n", err_new)
    @printf("Max abs difference: newautodiff vs old  = %.8e\n", diff)
    
    if diff < 1e-10
        println("Verification SUCCESS: Results are consistent.")
    else
        println("Verification FAILURE: Results still inconsistent.")
    end
end

verify_accuracy()
