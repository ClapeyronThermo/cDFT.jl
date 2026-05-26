using cDFT, Clapeyron, CUDA, Adapt, KernelAbstractions, LinearAlgebra

function verify_source()
    if !CUDA.functional(); println("CUDA not functional"); return; end
    
    # Setup a simple 1D hexane system
    model = PCSAFT(["hexane"])
    T = 298.15
    L = 20.0
    ngrid = (128,)
    structure = cDFT.Uniform1DCart((1.0e5, T), [0.1], [0.0, L], 128)
    options = cDFT.DFTOptions(CUDABackend())
    system = cDFT.DFTSystem(model, structure, options)
    
    ρ = fill(0.1, 128, 1)
    ρ_gpu = Adapt.adapt(CUDABackend(), ρ)
    
    println("Calling δFδρ_res_newautodiff (Source implementation)...")
    try
        # This will call our newly refactored Enzyme code
        dF = cDFT.δFδρ_res_newautodiff(system, ρ_gpu)
        KernelAbstractions.synchronize(CUDABackend())
        
        println("Functional derivative obtained successfully.")
        println("Sample values: ", Array(dF)[1:5, 1])
        
        if all(x -> x != 0.0, Array(dF))
            println("Verification SUCCESS: Non-zero gradients obtained.")
        else
            println("Verification FAILURE: Gradients are zero.")
        end
    catch e
        println("Verification FAILURE: Exception caught during execution.")
        showerror(stdout, e)
        println()
        rethrow(e)
    end
end

verify_source()
