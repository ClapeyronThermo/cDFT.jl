include("weighted_densities.jl")

"""
    evaluate_field(system::DFTSystem, profiles)

This function will obtain every field used in the system (listed in `system.fields`). The output will be a 3D array with the dimensions `(ngrid,nf,nc)`, where `ngrid` is the number of grid points, `nf` is the number of fields, and `nc` is the number of components in the model.

This is the macro function that will call `evaluate_field(system,field,profiles)` for each field in the system.
"""
function evaluate_field(system::DFTSystem, ρ)
    fields = system.fields
    nf = length(fields)
    nb = size(ρ,2)
    ngrid = system.structure.ngrid

    n = zeros(Float64,ngrid,nf,nb)

    for i in 1:nf
        n[:,i,:] = evaluate_field(system,fields[i], ρ)
    end
    
    return n
end

"""
    integrate_field(system::DFTSystem, δf, ρ)

This function will obtain, for all fields, the functional derivative for each species / bead. The output will be a 2D array with the dimensions `(ngrid,nb)`, where `ngrid` is the number of grid points, and `nb` is the number of beads in the model.

This is the macro function that will call `integrate_field(system,field,δf,ρ)` for each field in the system.
"""
function integrate_field(system::DFTSystem, δf, ρ)
    fields = system.fields

    ngrid = system.structure.ngrid
    nb  = size(ρ,2)
    nf = length(fields)

    δFδρ_res = zeros(ngrid,nb)

    for j in 1:nf        
        δFδρ_res += integrate_field(system,fields[j],@view(δf[:,j,:]))
    end

    return δFδρ_res
end