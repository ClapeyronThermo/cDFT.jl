include("weighted_densities.jl")

"""
    evaluate_field(system::DFTSystem)

This function will obtain every field used in the system (listed in `system.fields`). The output will be a 3D array with the dimensions `(ngrid,nf,nc)`, where `ngrid` is the number of grid points, `nf` is the number of fields, and `nc` is the number of components in the model.

This is the macro function that will call `evaluate_field(system,field)` for each field in the system.
"""
function evaluate_field!(system::DFTSystem)
    fields = system.fields
    nf = length(fields)
    fields = system.fields
    profiles = system.profiles
    species = system.species
    structure = system.structure

    for i in 1:nf
        system.cache.n[:,i,:] .= evaluate_field(system.cache.n[:,i,:],structure,fields[i],profiles,species)
    end
end

"""
    integrate_field(system::DFTSystem, δf)

This function will obtain, for all fields, the functional derivative for each species / bead. The output will be a 2D array with the dimensions `(ngrid,nc)`, where `ngrid` is the number of grid points, and `nc` is the number of components in the model.

This is the macro function that will call `integrate_field(system,field,δf)` for each field in the system.
"""
function integrate_field!(system::DFTSystem)
    profiles = system.profiles
    fields = system.fields
    species = system.species
    structure = system.structure

    nb  = length(profiles)
    nf = length(fields)


    for j in 1:nf        
        for i in 1:nb
            update_profile!(fields[j].profiles[i],system.cache.n[:,j,i])
        end
        system.cache.n[:,1,:] += integrate_field(system.cache.n[:,1,:],structure,fields[j],profiles,species)
    end

end