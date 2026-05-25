forwarddiff, limitations in GPU, and now not working in GPU yet

benchmark1: version now
work1: gpu forwarddiff
work2: gpu enzyme

@edit to see functions source code

[2] GPU Evaluation (δFδρ_res_GPU)                                                                                                                                           
   ERROR: LoadError: GPU compilation of MethodInstance for cDFT.gpu_δf_kernel!(::KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize,               
   KernelAbstractions.NDIteration.DynamicCheck, Nothing, CartesianIndices{1, Tuple{Base.OneTo{Int64}}}, KernelAbstractions.NDIteration.NDRange{1,                              
   KernelAbstractions.NDIteration.DynamicSize, KernelAbstractions.NDIteration.DynamicSize, CartesianIndices{1, Tuple{Base.OneTo{Int64}}}, CartesianIndices{1,                  
   Tuple{Base.OneTo{Int64}}}}}, ::CuDeviceArray{Float64, 3, 1}, ::CuDeviceArray{Float64, 3, 1}, ::cDFT.var"#f#preallocate_model##0"{DFTSystem{PCSAFT{BasicIdeal, Float64},     
   cDFT.PCSAFTSpecies, Uniform1DCart, Tuple{cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},    
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.VWeightedDensity{CuArray{ComplexF64, 3,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}}, Nothing, cDFT.IdealPropagator, DFTOptions{CUDABackend}, 7}},      
   ::Val{7}, ::Val{1}) failed                                                                                                                                                  
   KernelError: passing non-bitstype argument                                                                                                                                  
                                                                                                                                                                               
   Argument 5 to your kernel function is of type cDFT.var"#f#preallocate_model##0"{DFTSystem{PCSAFT{BasicIdeal, Float64}, cDFT.PCSAFTSpecies, Uniform1DCart,                   
   Tuple{cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},                              
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.VWeightedDensity{CUDACore.CuArray{ComplexF64, 3,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}}, Nothing, cDFT.IdealPropagator,                                    
   DFTOptions{CUDACore.CUDAKernels.CUDABackend}, 7}}, which is not a bitstype:                                                                                                 
     .system is of type DFTSystem{PCSAFT{BasicIdeal, Float64}, cDFT.PCSAFTSpecies, Uniform1DCart, Tuple{cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2,                  
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.VWeightedDensity{CUDACore.CuArray{ComplexF64, 3, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}}, Nothing, cDFT.IdealPropagator, DFTOptions{CUDACore.CUDAKernels.CUDABackend}, 7} which is not isbits.                                         
       .model is of type PCSAFT{BasicIdeal, Float64} which is not isbits.                                                                                                      
         .components is of type Vector{String} which is not isbits.                                                                                                            
           .ref is of type MemoryRef{String} which is not isbits.                                                                                                              
             .mem is of type Memory{String} which is not isbits.                                                                                                               
         .sites is of type SiteParam which is not isbits.                                                                                                                      
           .components is of type Vector{String} which is not isbits.                                                                                                          
             .ref is of type MemoryRef{String} which is not isbits.                                                                                                            
               .mem is of type Memory{String} which is not isbits.                                                                                                             
           .sites is of type Vector{Vector{String}} which is not isbits.                                                                                                       
             .ref is of type MemoryRef{Vector{String}} which is not isbits.                                                                                                    
               .mem is of type Memory{Vector{String}} which is not isbits.                                                                                                     
           .n_sites is of type PackedVectorsOfVectors.PackedVectorOfVectors{Vector{Int64}, Vector{Int64}, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}    
   which is not isbits.                                                                                                                                                        
             .p is of type Vector{Int64} which is not isbits.                                                                                                                  
               .ref is of type MemoryRef{Int64} which is not isbits.                                                                                                           
                 .mem is of type Memory{Int64} which is not isbits.                                                                                                            
             .v is of type Vector{Int64} which is not isbits.                                                                                                                  
               .ref is of type MemoryRef{Int64} which is not isbits.                                                                                                           
                 .mem is of type Memory{Int64} which is not isbits.                                                                                                            
           .i_sites is of type Vector{Vector{Int64}} which is not isbits.                                                                                                      
             .ref is of type MemoryRef{Vector{Int64}} which is not isbits.                                                                                                     
               .mem is of type Memory{Vector{Int64}} which is not isbits.                                                                                                      
           .flattenedsites is of type Vector{String} which is not isbits.                                                                                                      
             .ref is of type MemoryRef{String} which is not isbits.                                                                                                            
               .mem is of type Memory{String} which is not isbits.                                                                                                             
           .n_flattenedsites is of type Vector{Vector{Int64}} which is not isbits.                                                                                             
             .ref is of type MemoryRef{Vector{Int64}} which is not isbits.                                                                                                     
               .mem is of type Memory{Vector{Int64}} which is not isbits.                                                                                                      
           .i_flattenedsites is of type Vector{Vector{Int64}} which is not isbits.                                                                                             
             .ref is of type MemoryRef{Vector{Int64}} which is not isbits.                                                                                                     
               .mem is of type Memory{Vector{Int64}} which is not isbits.                                                                                                      
           .sourcecsvs is of type Vector{String} which is not isbits.                                                                                                          
             .ref is of type MemoryRef{String} which is not isbits.                                                                                                            
               .mem is of type Memory{String} which is not isbits.                                                                                                             
           .site_translator is of type Union{Nothing, Vector{Vector{Tuple{Int64, Int64}}}} which is not isbits.                                                                
         .params is of type Clapeyron.PCSAFTParam{Float64} which is not isbits.                                                                                                
           .Mw is of type SingleParam{Float64} which is not isbits.                                                                                                            
             .name is of type String which is not isbits.                                                                                                                      
             .components is of type Vector{String} which is not isbits.                                                                                                        
               .ref is of type MemoryRef{String} which is not isbits.                                                                                                          
                 .mem is of type Memory{String} which is not isbits.                                                                                                           
             .values is of type Vector{Float64} which is not isbits.                                                                                                           
               .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                         
                 .mem is of type Memory{Float64} which is not isbits.                                                                                                          
             .ismissingvalues is of type Vector{Bool} which is not isbits.                                                                                                     
               .ref is of type MemoryRef{Bool} which is not isbits.                                                                                                            
                 .mem is of type Memory{Bool} which is not isbits.                                                                                                             
             .sourcecsvs is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                        
             .sources is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                           
           .segment is of type SingleParam{Float64} which is not isbits.                                                                                                       
             .name is of type String which is not isbits.                                                                                                                      
             .components is of type Vector{String} which is not isbits.                                                                                                        
               .ref is of type MemoryRef{String} which is not isbits.                                                                                                          
                 .mem is of type Memory{String} which is not isbits.                                                                                                           
             .values is of type Vector{Float64} which is not isbits.                                                                                                           
               .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                         
                 .mem is of type Memory{Float64} which is not isbits.                                                                                                          
             .ismissingvalues is of type Vector{Bool} which is not isbits.                                                                                                     
               .ref is of type MemoryRef{Bool} which is not isbits.                                                                                                            
                 .mem is of type Memory{Bool} which is not isbits.                                                                                                             
             .sourcecsvs is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                        
             .sources is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                           
           .sigma is of type PairParam{Float64} which is not isbits.                                                                                                           
             .name is of type String which is not isbits.                                                                                                                      
             .components is of type Vector{String} which is not isbits.                                                                                                        
               .ref is of type MemoryRef{String} which is not isbits.                                                                                                          
                 .mem is of type Memory{String} which is not isbits.                                                                                                           
             .values is of type Matrix{Float64} which is not isbits.                                                                                                           
               .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                         
                 .mem is of type Memory{Float64} which is not isbits.                                                                                                          
             .ismissingvalues is of type Matrix{Bool} which is not isbits.                                                                                                     
               .ref is of type MemoryRef{Bool} which is not isbits.                                                                                                            
                 .mem is of type Memory{Bool} which is not isbits.                                                                                                             
             .sourcecsvs is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                        
             .sources is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                           
           .epsilon is of type PairParam{Float64} which is not isbits.                                                                                                         
             .name is of type String which is not isbits.                                                                                                                      
             .components is of type Vector{String} which is not isbits.                                                                                                        
               .ref is of type MemoryRef{String} which is not isbits.                                                                                                          
                 .mem is of type Memory{String} which is not isbits.                                                                                                           
             .values is of type Matrix{Float64} which is not isbits.                                                                                                           
               .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                         
                 .mem is of type Memory{Float64} which is not isbits.                                                                                                          
             .ismissingvalues is of type Matrix{Bool} which is not isbits.                                                                                                     
               .ref is of type MemoryRef{Bool} which is not isbits.                                                                                                            
                 .mem is of type Memory{Bool} which is not isbits.                                                                                                             
             .sourcecsvs is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                        
             .sources is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                           
           .epsilon_assoc is of type AssocParam{Float64} which is not isbits.                                                                                                  
             .name is of type String which is not isbits.                                                                                                                      
             .components is of type Vector{String} which is not isbits.                                                                                                        
               .ref is of type MemoryRef{String} which is not isbits.                                                                                                          
                 .mem is of type Memory{String} which is not isbits.                                                                                                           
             .values is of type Clapeyron.Compressed4DMatrix{Float64, Vector{Float64}} which is not isbits.                                                                    
               .values is of type Vector{Float64} which is not isbits.                                                                                                         
                 .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                       
                   .mem is of type Memory{Float64} which is not isbits.                                                                                                        
               .outer_indices is of type Vector{Tuple{Int64, Int64}} which is not isbits.                                                                                      
                 .ref is of type MemoryRef{Tuple{Int64, Int64}} which is not isbits.                                                                                           
                   .mem is of type Memory{Tuple{Int64, Int64}} which is not isbits.                                                                                            
               .inner_indices is of type Vector{Tuple{Int64, Int64}} which is not isbits.                                                                                      
                 .ref is of type MemoryRef{Tuple{Int64, Int64}} which is not isbits.                                                                                           
                   .mem is of type Memory{Tuple{Int64, Int64}} which is not isbits.                                                                                            
             .sites is of type Union{Nothing, Vector{Vector{String}}} which is not isbits.                                                                                     
             .sourcecsvs is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                        
             .sources is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                           
           .bondvol is of type AssocParam{Float64} which is not isbits.                                                                                                        
             .name is of type String which is not isbits.                                                                                                                      
             .components is of type Vector{String} which is not isbits.                                                                                                        
               .ref is of type MemoryRef{String} which is not isbits.                                                                                                          
                 .mem is of type Memory{String} which is not isbits.                                                                                                           
             .values is of type Clapeyron.Compressed4DMatrix{Float64, Vector{Float64}} which is not isbits.                                                                    
               .values is of type Vector{Float64} which is not isbits.                                                                                                         
                 .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                       
                   .mem is of type Memory{Float64} which is not isbits.                                                                                                        
               .outer_indices is of type Vector{Tuple{Int64, Int64}} which is not isbits.                                                                                      
                 .ref is of type MemoryRef{Tuple{Int64, Int64}} which is not isbits.                                                                                           
                   .mem is of type Memory{Tuple{Int64, Int64}} which is not isbits.                                                                                            
               .inner_indices is of type Vector{Tuple{Int64, Int64}} which is not isbits.                                                                                      
                 .ref is of type MemoryRef{Tuple{Int64, Int64}} which is not isbits.                                                                                           
                   .mem is of type Memory{Tuple{Int64, Int64}} which is not isbits.                                                                                            
             .sites is of type Union{Nothing, Vector{Vector{String}}} which is not isbits.                                                                                     
             .sourcecsvs is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                        
             .sources is of type Union{Nothing, Vector{String}} which is not isbits.                                                                                           
         .assoc_options is of type AssocOptions which is not isbits.                                                                                                           
           .combining is of type Symbol which is not isbits.                                                                                                                   
         .references is of type Vector{String} which is not isbits.                                                                                                            
           .ref is of type MemoryRef{String} which is not isbits.                                                                                                              
             .mem is of type Memory{String} which is not isbits.                                                                                                               
       .species is of type cDFT.PCSAFTSpecies which is not isbits.                                                                                                             
         .nbeads is of type Vector{Int64} which is not isbits.                                                                                                                 
           .ref is of type MemoryRef{Int64} which is not isbits.                                                                                                               
             .mem is of type Memory{Int64} which is not isbits.                                                                                                                
         .size is of type Vector{Float64} which is not isbits.                                                                                                                 
           .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                             
             .mem is of type Memory{Float64} which is not isbits.                                                                                                              
         .bulk_density is of type Vector{Float64} which is not isbits.                                                                                                         
           .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                             
             .mem is of type Memory{Float64} which is not isbits.                                                                                                              
         .chempot_res is of type Vector{Float64} which is not isbits.                                                                                                          
           .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                             
             .mem is of type Memory{Float64} which is not isbits.                                                                                                              
       .structure is of type Uniform1DCart which is not isbits.                                                                                                                
         .ρbulk is of type Vector{Float64} which is not isbits.                                                                                                                
           .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                             
             .mem is of type Memory{Float64} which is not isbits.                                                                                                              
         .bounds is of type Vector{Float64} which is not isbits.                                                                                                               
           .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                             
             .mem is of type Memory{Float64} which is not isbits.                                                                                                              
       .fields is of type Tuple{cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},       
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.VWeightedDensity{CUDACore.CuArray{ComplexF64, 3,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2,              
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},      
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}} which is not isbits.                                               
         .1 is of type cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},                
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}} which is not isbits.                                                
           .type is of type Symbol which is not isbits.                                                                                                                        
           .width is of type Vector{Float64} which is not isbits.                                                                                                              
             .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                           
               .mem is of type Memory{Float64} which is not isbits.                                                                                                            
           .map is of type CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory} which is not isbits.                                                                         
             .data is of type GPUArrays.DataRef{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                                  
               .rc is of type GPUArrays.RefCounted{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                               
                 .obj is of type CUDACore.Managed{CUDACore.DeviceMemory} which is not isbits.                                                                                  
                   .stream is of type CUDACore.CuStream which is not isbits.                                                                                                   
                     .ctx is of type Union{Nothing, CUDACore.CuContext} which is not isbits.                                                                                   
                 .finalizer is of type Any which is not isbits.                                                                                                                
                 .count is of type Base.Threads.Atomic{Int64} which is not isbits.                                                                                             
           .plan is of type FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}} which is not isbits.                                                                     
             .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                     
               .p is of type Any which is not isbits.                                                                                                                          
               .scale is of type Any which is not isbits.                                                                                                                      
           .iplan is of type AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64} which is not isbits.                       
             .p is of type FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}} which is not isbits.                                                                       
               .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                   
                 .p is of type Any which is not isbits.                                                                                                                        
                 .scale is of type Any which is not isbits.                                                                                                                    
         .2 is of type cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},                
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}} which is not isbits.                                                
           .type is of type Symbol which is not isbits.                                                                                                                        
           .width is of type Vector{Float64} which is not isbits.                                                                                                              
             .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                           
               .mem is of type Memory{Float64} which is not isbits.                                                                                                            
           .map is of type CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory} which is not isbits.                                                                         
             .data is of type GPUArrays.DataRef{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                                  
               .rc is of type GPUArrays.RefCounted{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                               
                 .obj is of type CUDACore.Managed{CUDACore.DeviceMemory} which is not isbits.                                                                                  
                   .stream is of type CUDACore.CuStream which is not isbits.                                                                                                   
                     .ctx is of type Union{Nothing, CUDACore.CuContext} which is not isbits.                                                                                   
                 .finalizer is of type Any which is not isbits.                                                                                                                
                 .count is of type Base.Threads.Atomic{Int64} which is not isbits.                                                                                             
           .plan is of type FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}} which is not isbits.                                                                     
             .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                     
               .p is of type Any which is not isbits.                                                                                                                          
               .scale is of type Any which is not isbits.                                                                                                                      
           .iplan is of type AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64} which is not isbits.                       
             .p is of type FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}} which is not isbits.                                                                       
               .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                   
                 .p is of type Any which is not isbits.                                                                                                                        
                 .scale is of type Any which is not isbits.                                                                                                                    
         .3 is of type cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},                
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}} which is not isbits.                                                
           .type is of type Symbol which is not isbits.                                                                                                                        
           .width is of type Vector{Float64} which is not isbits.                                                                                                              
             .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                           
               .mem is of type Memory{Float64} which is not isbits.                                                                                                            
           .map is of type CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory} which is not isbits.                                                                         
             .data is of type GPUArrays.DataRef{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                                  
               .rc is of type GPUArrays.RefCounted{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                               
                 .obj is of type CUDACore.Managed{CUDACore.DeviceMemory} which is not isbits.                                                                                  
                   .stream is of type CUDACore.CuStream which is not isbits.                                                                                                   
                     .ctx is of type Union{Nothing, CUDACore.CuContext} which is not isbits.                                                                                   
                 .finalizer is of type Any which is not isbits.                                                                                                                
                 .count is of type Base.Threads.Atomic{Int64} which is not isbits.                                                                                             
           .plan is of type FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}} which is not isbits.                                                                     
             .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                     
               .p is of type Any which is not isbits.                                                                                                                          
               .scale is of type Any which is not isbits.                                                                                                                      
           .iplan is of type AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64} which is not isbits.                       
             .p is of type FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}} which is not isbits.                                                                       
               .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                   
                 .p is of type Any which is not isbits.                                                                                                                        
                 .scale is of type Any which is not isbits.                                                                                                                    
         .4 is of type cDFT.VWeightedDensity{CUDACore.CuArray{ComplexF64, 3, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},                
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}} which is not isbits.                                                
           .type is of type Symbol which is not isbits.                                                                                                                        
           .width is of type Vector{Float64} which is not isbits.                                                                                                              
             .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                           
               .mem is of type Memory{Float64} which is not isbits.                                                                                                            
           .map is of type CUDACore.CuArray{ComplexF64, 3, CUDACore.DeviceMemory} which is not isbits.                                                                         
             .data is of type GPUArrays.DataRef{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                                  
               .rc is of type GPUArrays.RefCounted{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                               
                 .obj is of type CUDACore.Managed{CUDACore.DeviceMemory} which is not isbits.                                                                                  
                   .stream is of type CUDACore.CuStream which is not isbits.                                                                                                   
                     .ctx is of type Union{Nothing, CUDACore.CuContext} which is not isbits.                                                                                   
                 .finalizer is of type Any which is not isbits.                                                                                                                
                 .count is of type Base.Threads.Atomic{Int64} which is not isbits.                                                                                             
           .plan is of type FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}} which is not isbits.                                                                     
             .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                     
               .p is of type Any which is not isbits.                                                                                                                          
               .scale is of type Any which is not isbits.                                                                                                                      
           .iplan is of type AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64} which is not isbits.                       
             .p is of type FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}} which is not isbits.                                                                       
               .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                   
                 .p is of type Any which is not isbits.                                                                                                                        
                 .scale is of type Any which is not isbits.                                                                                                                    
         .5 is of type cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},                
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}} which is not isbits.                                                
           .type is of type Symbol which is not isbits.                                                                                                                        
           .width is of type Vector{Float64} which is not isbits.                                                                                                              
             .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                           
               .mem is of type Memory{Float64} which is not isbits.                                                                                                            
           .map is of type CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory} which is not isbits.                                                                         
             .data is of type GPUArrays.DataRef{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                                  
               .rc is of type GPUArrays.RefCounted{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                               
                 .obj is of type CUDACore.Managed{CUDACore.DeviceMemory} which is not isbits.                                                                                  
                   .stream is of type CUDACore.CuStream which is not isbits.                                                                                                   
                     .ctx is of type Union{Nothing, CUDACore.CuContext} which is not isbits.                                                                                   
                 .finalizer is of type Any which is not isbits.                                                                                                                
                 .count is of type Base.Threads.Atomic{Int64} which is not isbits.                                                                                             
           .plan is of type FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}} which is not isbits.                                                                     
             .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                     
               .p is of type Any which is not isbits.                                                                                                                          
               .scale is of type Any which is not isbits.                                                                                                                      
           .iplan is of type AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64} which is not isbits.                       
             .p is of type FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}} which is not isbits.                                                                       
               .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                   
                 .p is of type Any which is not isbits.                                                                                                                        
                 .scale is of type Any which is not isbits.                                                                                                                    
         .6 is of type cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},                
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}} which is not isbits.                                                
           .type is of type Symbol which is not isbits.                                                                                                                        
           .width is of type Vector{Float64} which is not isbits.                                                                                                              
             .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                           
               .mem is of type Memory{Float64} which is not isbits.                                                                                                            
           .map is of type CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory} which is not isbits.                                                                         
             .data is of type GPUArrays.DataRef{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                                  
               .rc is of type GPUArrays.RefCounted{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                               
                 .obj is of type CUDACore.Managed{CUDACore.DeviceMemory} which is not isbits.                                                                                  
                   .stream is of type CUDACore.CuStream which is not isbits.                                                                                                   
                     .ctx is of type Union{Nothing, CUDACore.CuContext} which is not isbits.                                                                                   
                 .finalizer is of type Any which is not isbits.                                                                                                                
                 .count is of type Base.Threads.Atomic{Int64} which is not isbits.                                                                                             
           .plan is of type FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}} which is not isbits.                                                                     
             .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                     
               .p is of type Any which is not isbits.                                                                                                                          
               .scale is of type Any which is not isbits.                                                                                                                      
           .iplan is of type AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64} which is not isbits.                       
             .p is of type FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}} which is not isbits.                                                                       
               .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                   
                 .p is of type Any which is not isbits.                                                                                                                        
                 .scale is of type Any which is not isbits.                                                                                                                    
         .7 is of type cDFT.SWeightedDensity{CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},                
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}} which is not isbits.                                                
           .type is of type Symbol which is not isbits.                                                                                                                        
           .width is of type Vector{Float64} which is not isbits.                                                                                                              
             .ref is of type MemoryRef{Float64} which is not isbits.                                                                                                           
               .mem is of type Memory{Float64} which is not isbits.                                                                                                            
           .map is of type CUDACore.CuArray{ComplexF64, 2, CUDACore.DeviceMemory} which is not isbits.                                                                         
             .data is of type GPUArrays.DataRef{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                                  
               .rc is of type GPUArrays.RefCounted{CUDACore.Managed{CUDACore.DeviceMemory}} which is not isbits.                                                               
                 .obj is of type CUDACore.Managed{CUDACore.DeviceMemory} which is not isbits.                                                                                  
                   .stream is of type CUDACore.CuStream which is not isbits.                                                                                                   
                     .ctx is of type Union{Nothing, CUDACore.CuContext} which is not isbits.                                                                                   
                 .finalizer is of type Any which is not isbits.                                                                                                                
                 .count is of type Base.Threads.Atomic{Int64} which is not isbits.                                                                                             
           .plan is of type FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}} which is not isbits.                                                                     
             .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                     
               .p is of type Any which is not isbits.                                                                                                                          
               .scale is of type Any which is not isbits.                                                                                                                      
           .iplan is of type AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64} which is not isbits.                       
             .p is of type FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}} which is not isbits.                                                                       
               .pinv is of type AbstractFFTs.ScaledPlan which is not isbits.                                                                                                   
                 .p is of type Any which is not isbits.                                                                                                                        
                 .scale is of type Any which is not isbits.                                                                                                                    
                                                                                                                                                                               
                                                                                                                                                                               
   Only bitstypes, which are "plain data" types that are immutable                                                                                                             
   and contain no references to other values, can be used in GPU kernels.                                                                                                      
   For more information, see the `Base.isbitstype` function.                                                                                                                   
                                                                                                                                                                               
   Stacktrace:                                                                                                                                                                 
     [1] check_invocation(job::GPUCompiler.CompilerJob)                                                                                                                        
       @ GPUCompiler ~/.julia/packages/GPUCompiler/BuBOo/src/validation.jl:108                                                                                                 
     [2] compile_unhooked(output::Symbol, job::GPUCompiler.CompilerJob; kwargs::@Kwargs{})                                                                                     
       @ GPUCompiler ~/.julia/packages/GPUCompiler/BuBOo/src/driver.jl:87                                                                                                      
     [3] compile_unhooked                                                                                                                                                      
       @ ~/.julia/packages/GPUCompiler/BuBOo/src/driver.jl:80 [inlined]                                                                                                        
     [4] #compile#96                                                                                                                                                           
       @ ~/.julia/packages/GPUCompiler/BuBOo/src/driver.jl:67 [inlined]                                                                                                        
     [5] compile(target::Symbol, job::GPUCompiler.CompilerJob)                                                                                                                 
       @ GPUCompiler ~/.julia/packages/GPUCompiler/BuBOo/src/driver.jl:55                                                                                                      
     [6] #invoke_frozen#583                                                                                                                                                    
       @ ~/.julia/packages/CUDACore/sIGRL/src/initialization.jl:30 [inlined]                                                                                                   
     [7] invoke_frozen                                                                                                                                                         
       @ ~/.julia/packages/CUDACore/sIGRL/src/initialization.jl:26 [inlined]                                                                                                   
     [8] #compile##0                                                                                                                                                           
       @ ~/.julia/packages/CUDACore/sIGRL/src/compiler/compilation.jl:255 [inlined]                                                                                            
     [9] JuliaContext(f::CUDACore.var"#compile##0#compile##1"{GPUCompiler.CompilerJob{GPUCompiler.PTXCompilerTarget, CUDACore.CUDACompilerParams}}; kwargs::@Kwargs{})         
       @ GPUCompiler ~/.julia/packages/GPUCompiler/BuBOo/src/driver.jl:34                                                                                                      
    [10] JuliaContext(f::Function)                                                                                                                                             
       @ GPUCompiler ~/.julia/packages/GPUCompiler/BuBOo/src/driver.jl:25                                                                                                      
    [11] compile(job::GPUCompiler.CompilerJob)                                                                                                                                 
       @ CUDACore ~/.julia/packages/CUDACore/sIGRL/src/compiler/compilation.jl:254                                                                                             
    [12] actual_compilation(cache::Dict{Any, CuFunction}, src::Core.MethodInstance, world::UInt64, cfg::GPUCompiler.CompilerConfig{GPUCompiler.PTXCompilerTarget,              
   CUDACore.CUDACompilerParams}, compiler::typeof(CUDACore.compile), linker::typeof(CUDACore.link))                                                                            
       @ GPUCompiler ~/.julia/packages/GPUCompiler/BuBOo/src/execution.jl:245                                                                                                  
    [13] cached_compilation(cache::Dict{Any, CuFunction}, src::Core.MethodInstance, cfg::GPUCompiler.CompilerConfig{GPUCompiler.PTXCompilerTarget,                             
   CUDACore.CUDACompilerParams}, compiler::Function, linker::Function)                                                                                                         
       @ GPUCompiler ~/.julia/packages/GPUCompiler/BuBOo/src/execution.jl:159                                                                                                  
    [14] macro expansion                                                                                                                                                       
       @ ~/.julia/packages/CUDACore/sIGRL/src/compiler/execution.jl:450 [inlined]                                                                                              
    [15] macro expansion                                                                                                                                                       
       @ ./lock.jl:376 [inlined]                                                                                                                                               
    [16] cufunction(f::typeof(cDFT.gpu_δf_kernel!), tt::Type{Tuple{KernelAbstractions.CompilerMetadata{KernelAbstractions.NDIteration.DynamicSize,                             
   KernelAbstractions.NDIteration.DynamicCheck, Nothing, CartesianIndices{1, Tuple{Base.OneTo{Int64}}}, KernelAbstractions.NDIteration.NDRange{1,                              
   KernelAbstractions.NDIteration.DynamicSize, KernelAbstractions.NDIteration.DynamicSize, CartesianIndices{1, Tuple{Base.OneTo{Int64}}}, CartesianIndices{1,                  
   Tuple{Base.OneTo{Int64}}}}}, CuDeviceArray{Float64, 3, 1}, CuDeviceArray{Float64, 3, 1}, cDFT.var"#f#preallocate_model##0"{DFTSystem{PCSAFT{BasicIdeal, Float64},           
   cDFT.PCSAFTSpecies, Uniform1DCart, Tuple{cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},    
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.VWeightedDensity{CuArray{ComplexF64, 3,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}}, Nothing, cDFT.IdealPropagator, DFTOptions{CUDABackend}, 7}},      
   Val{7}, Val{1}}}; kwargs::@Kwargs{always_inline::Bool, maxthreads::Nothing})                                                                                                
       @ CUDACore ~/.julia/packages/CUDACore/sIGRL/src/compiler/execution.jl:445                                                                                               
    [17] cufunction                                                                                                                                                            
       @ ~/.julia/packages/CUDACore/sIGRL/src/compiler/execution.jl:442 [inlined]                                                                                              
    [18] #kernel_compile#734                                                                                                                                                   
       @ ~/.julia/packages/CUDACore/sIGRL/src/compiler/execution.jl:59 [inlined]                                                                                               
    [19] macro expansion                                                                                                                                                       
       @ ~/.julia/packages/CUDACore/sIGRL/src/compiler/execution.jl:182 [inlined]                                                                                              
    [20] (::KernelAbstractions.Kernel{CUDABackend, KernelAbstractions.NDIteration.DynamicSize, KernelAbstractions.NDIteration.DynamicSize,                                     
   typeof(cDFT.gpu_δf_kernel!)})(::CuArray{Float64, 3, CUDACore.DeviceMemory}, ::Vararg{Any}; ndrange::Tuple{Int64}, workgroupsize::Nothing)                                   
       @ CUDACore.CUDAKernels ~/.julia/packages/CUDACore/sIGRL/src/CUDAKernels.jl:125                                                                                          
    [21] Kernel                                                                                                                                                                
       @ ~/.julia/packages/CUDACore/sIGRL/src/CUDAKernels.jl:111 [inlined]                                                                                                     
    [22] δFδρ_res_GPU!(system::DFTSystem{PCSAFT{BasicIdeal, Float64}, cDFT.PCSAFTSpecies, Uniform1DCart, Tuple{cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                    
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.VWeightedDensity{CuArray{ComplexF64, 3, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}}, Nothing, cDFT.IdealPropagator, DFTOptions{CUDABackend}, 7}, ρ::CuArray{Float64, 2, CUDACore.DeviceMemory}, δfδρ_res::CuArray{Float64, 2,     
   CUDACore.DeviceMemory}, n::CuArray{Float64, 3, CUDACore.DeviceMemory}, δf::CuArray{Float64, 3, CUDACore.DeviceMemory}, fft_buf::CuArray{Float64, 3, CUDACore.DeviceMemory}, 
   in_buf::CuArray{ComplexF64, 1, CUDACore.DeviceMemory}, out_buf::CuArray{ComplexF64, 1, CUDACore.DeviceMemory}, P::cuFFT.CuFFTPlan{ComplexF64, ComplexF64, -1, true, 1, 1,   
   Nothing}, iP::AbstractFFTs.ScaledPlan{ComplexF64, cuFFT.CuFFTPlan{ComplexF64, ComplexF64, 1, true, 1, 1, Nothing}, Float64},                                                
   f::cDFT.var"#f#preallocate_model##0"{DFTSystem{PCSAFT{BasicIdeal, Float64}, cDFT.PCSAFTSpecies, Uniform1DCart, Tuple{cDFT.SWeightedDensity{CuArray{ComplexF64, 2,           
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.VWeightedDensity{CuArray{ComplexF64, 3, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}}, Nothing, cDFT.IdealPropagator, DFTOptions{CUDABackend}, 7}}, cache_pool::Nothing)                                                            
       @ cDFT /mnt/d/Aho/Vibe_Project/cDFT/cDFT.jl/src/models/models.jl:132                                                                                                    
    [23] δFδρ_res_GPU(system::DFTSystem{PCSAFT{BasicIdeal, Float64}, cDFT.PCSAFTSpecies, Uniform1DCart, Tuple{cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                     
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.VWeightedDensity{CuArray{ComplexF64, 3, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2, CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}},               
   AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1, UnitRange{Int64}}, Float64}}, cDFT.SWeightedDensity{CuArray{ComplexF64, 2,                       
   CUDACore.DeviceMemory}, FFTW.cFFTWPlan{ComplexF64, -1, true, 1, UnitRange{Int64}}, AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, true, 1,               
   UnitRange{Int64}}, Float64}}}, Nothing, cDFT.IdealPropagator, DFTOptions{CUDABackend}, 7}, ρ::CuArray{Float64, 2, CUDACore.DeviceMemory})                                   
       @ cDFT /mnt/d/Aho/Vibe_Project/cDFT/cDFT.jl/src/models/models.jl:143                                                                                                    
    [24] top-level scope                                                                                                                                                       
       @ /mnt/d/Aho/Vibe_Project/cDFT/cDFT.jl/benchmark/compare_minimal.jl:71                                                                                                  
    [25] include(mod::Module, _path::String)                                                                                                                                   
       @ Base ./Base.jl:306                                                                                                                                                    
    [26] exec_options(opts::Base.JLOptions)                                                                                                                                    
       @ Base ./client.jl:317                                                                                                                                                  
    [27] _start()                                                                                                                                                              
       @ Base ./client.jl:550                                                                                                                                                  
   in expression starting at /mnt/d/Aho/Vibe_Project/cDFT/cDFT.jl/benchmark/compare_minimal.jl:65                                                                              
   Exit Code: 1  


Might need to do:

gpu_params = extract_to_cuarrays(system.model) # 提前提取

f(x) = f_res_gpu(gpu_params, T, x) 

function f_res_gpu(params, T, x)
    # todo
end