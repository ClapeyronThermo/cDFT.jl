
function propagate(system::DFTSystem, propagate::TangentHSPropagator, δf_res)
    ρ = system.profiles
    model = system.model
    structure = system.structure
    ngrid = structure.ngrid
    species = system.species
    nbeads = sum(system.species.nbeads)
    z = system.profiles[1].coords

    Gcα = ones(Float64, ngrid, nbeads, nbeads)
    Gp  = ones(Float64, ngrid, nbeads)

    levels = species.levels
    for i in @comps
        if system.species.nbeads[i] !== 1
            n_intergroups = model.groups.n_intergroups[i] .== 1
            i_groups = model.groups.i_groups[i]
            # Get the levels
            n_levels = maximum(levels[i_groups])

            i_root = i_groups[levels[i_groups].==1][1]
            is_leaf = sum(n_intergroups,dims=1).==1 .&& (levels.!=1)'
            # Get Gαk
            for L in n_levels:-1:1
                i_group_level = i_groups[findall(levels[i_groups].==L)]
                for k in i_group_level
                    k_children = findall(n_intergroups[k,:] .&& levels.==L+1)
                    if !is_leaf[k]
                        for α in k_children
                            β = findall(n_intergroups[α,:] .&& levels.==L+2)
                            if isempty(β)
                                _Gcα = @. exp(-δf_res[:,α])
                            else
                                _Gcα = exp.(-δf_res[:,α]).*prod(Gcα[:,α,β],dims=(2,3))
                            end

                            lim = (system.species.size[k]+system.species.size[α])/2
                            bounds = system.structure.bounds.+(-lim,lim)
                            boundary_conditions = ρ[k].boundary_conditions
                            bc1 = typeof(boundary_conditions[1])(_Gcα[1],-1)
                            bc2 = typeof(boundary_conditions[2])(_Gcα[end],1)
                            _Gcα = DensityProfile(_Gcα,z,bounds,(bc1,bc2))
                            for j in 1:ngrid
                                Gcα[j,k,α] = ∫ρdz(structure, _Gcα, z[j], lim)./(2*lim)
                            end
                        end
                    end
                end
            end

            # Get I2
            for L in 1:n_levels
                i_group_level = i_groups[findall(levels[i_groups].==L)]
                for k in i_group_level
                    if k == i_root
                        Gp[:,k] .= 1.
                    else
                        l = findall(n_intergroups[k,:] .&& levels.==L-1)[1]

                        α = findall(n_intergroups[l,:] .&& levels.==L)
                        α = α[α.!=k]
                    
                        _Gp = exp.(-δf_res[:,l]).*Gp[:,l].*prod(Gcα[:,l,α],dims=(2,3))

                        lim = (system.species.size[k]+system.species.size[l])/2
                        bounds = system.structure.bounds.+(-lim,lim)
                        boundary_conditions = ρ[k].boundary_conditions
                        bc1 = typeof(boundary_conditions[1])(_Gp[1],-1)
                        bc2 = typeof(boundary_conditions[2])(_Gp[end],1)
                        _Gp = DensityProfile(_Gp,z,bounds,(bc1,bc2))
                        for j in 1:ngrid
                            Gp[j,k] = ∫ρdz(structure, _Gp, z[j], lim)./(2*lim)
                        end
                    end
                end
            end
        end
    end

    return Gcα, Gp
end

export converge!