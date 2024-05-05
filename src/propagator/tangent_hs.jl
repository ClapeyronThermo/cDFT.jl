
function propagate(system::DFTSystem, propagate::TangentHSPropagator, δf_res, species_id)
    ρ = system.profiles
    structure = system.structure
    ngrid = structure.ngrid
    nbeads = system.species[species_id].nbeads
    connectivity = system.species[species_id].connectivity
    z = system.profiles[1].coords

    I1 = ones(Float64, ngrid, nbeads)
    I2 = ones(Float64, ngrid, nbeads)

    leaves = findall(sum(connectivity; dims=1).==1)
    id1_1 = leaves[1][2]
    id1_2 = findfirst(connectivity[:,id1_1].==1)
    id2_1 = leaves[2][2]
    id2_2 = findfirst(connectivity[:,id2_1].==1)

    for i in 2:nbeads
        _I1 = @. I1[:,id1_1]*exp(-δf_res[:,id1_1])

        lim = (system.species[species_id].size[id1_1]+system.species[species_id].size[id1_2])/2
        bounds = system.structure.bounds.+(-lim,lim)
        boundary_conditions = ρ[i].boundary_conditions
        bc1 = typeof(boundary_conditions[1])(_I1[1],-1)
        bc2 = typeof(boundary_conditions[2])(_I1[end],1)

        _I1 = DensityProfile(_I1,z,bounds,(bc1,bc2))
        
        for j in 1:ngrid
            I1[j,id1_2] = ∫ρdz(structure, _I1, z[j], lim)./(2*lim)
        end

        id1_next = findall(connectivity[:,id1_2].==1)
        if length(id1_next) > 1
            id1_next = id1_next[id1_next.!=id1_1]
            id1_1 = deepcopy(id1_2)
            id1_2 = deepcopy(id1_next[1])
        end

        _I2 = @. I2[:,id2_1]*exp(-δf_res[:,id2_1])

        lim = (system.species[species_id].size[id2_1]+system.species[species_id].size[id2_2])/2
        bounds = system.structure.bounds.+(-lim,lim)
        boundary_conditions = ρ[i].boundary_conditions
        bc1 = typeof(boundary_conditions[1])(_I2[1],-1)
        bc2 = typeof(boundary_conditions[2])(_I2[end],1)

        _I2 = DensityProfile(_I2,z,bounds,(bc1,bc2))
        
        for j in 1:ngrid
            I2[j,id2_2] = ∫ρdz(structure, _I2, z[j], lim)./(2*lim)
        end

        id2_next = findall(connectivity[:,id2_2].==1)
        if length(id2_next) > 1
            id2_next = id2_next[id2_next.!=id2_1]
            id2_1 = deepcopy(id2_2)
            id2_2 = deepcopy(id2_next[1])
        end
    end

    return I1, I2
end

export converge!