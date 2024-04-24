include("weighted_densities.jl")

"""
    evaluate_field(system::DFTSystem)

This function will obtain every field used in the system (listed in `system.fields`). The output will be a 3D array with the dimensions `(ngrid,nf,nc)`, where `ngrid` is the number of grid points, `nf` is the number of fields, and `nc` is the number of components in the model.

This is the macro function that will call `evaluate_field(system,field)` for each field in the system.
"""
function evaluate_field(system::DFTSystem)
    fields = system.fields
    nf = length(fields)
    nc = length(system.model)
    ngrid = system.structure.ngrid

    n = zeros(ngrid,nf,nc)

    for i in 1:nf
        n[:,i,:] = evaluate_field(system,fields[i])
    end
    
    return n
end

