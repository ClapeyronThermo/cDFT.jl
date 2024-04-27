include("weighted_densities.jl")

"""
    evaluate_field(system::DFTSystem)

This function will obtain every field used in the system (listed in `system.fields`). The output will be a 3D array with the dimensions `(ngrid,nf,nc)`, where `ngrid` is the number of grid points, `nf` is the number of fields, and `nc` is the number of components in the model.

This is the macro function that will call `evaluate_field(system,field)` for each field in the system.
"""
function evaluate_field(system::DFTSystem)
    fields = system.fields
    nf = length(fields)
    nb = length(system.profiles)
    ngrid = system.structure.ngrid

    n = zeros(ngrid,nf,nb)

    for i in 1:nf
        n[:,i,:] = evaluate_field(system,fields[i])
    end
    
    return n
end

"""
    integrate_field(system::DFTSystem, δf)

This function will obtain, for all fields, the functional derivative for each species / bead. The output will be a 2D array with the dimensions `(ngrid,nc)`, where `ngrid` is the number of grid points, and `nc` is the number of components in the model.

This is the macro function that will call `integrate_field(system,field,δf)` for each field in the system.
"""
function integrate_field(system::DFTSystem, δf)
    ρ = system.profiles
    fields = system.fields
    species = system.species

    z = ρ[1].coords
    ngrid = system.structure.ngrid
    nb  = length(ρ)
    nf = length(fields)

    δFδρ_res = zeros(ngrid,nb)

    for j in 1:nf
        ∂f = DensityProfile[]
        
        species_id = 1
        bead_id = 1
        for i in 1:nb
            lim = species[species_id].size[bead_id]
            bounds = ρ[i].bounds.+(-lim,lim)
            boundary_conditions = ρ[i].boundary_conditions
            bc1 = typeof(boundary_conditions[1])(δf[1,j,i],-1)
            bc2 = typeof(boundary_conditions[2])(δf[end,j,i],1)
            push!(∂f, DensityProfile(@view(δf[:,j,i]),z,bounds,(bc1,bc2)))

            if bead_id == species[species_id].nbeads
                species_id += 1
                bead_id = 1
            else
                bead_id += 1
            end
        end
        δFδρ_res += integrate_field(system,fields[j],∂f)
    end

    return δFδρ_res
end