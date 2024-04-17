include("weighted_densities.jl")

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

