function TangentHSPropagator(model::EoSModel,species::DFTSpecies,structure::DFTStructure)
    ngrid = structure.ngrid
    nbeads = sum(species.nbeads)
    map = zeros(ComplexF64, ngrid, nbeads, nbeads)
    f = structure.ngrid/(structure.bounds[2]-structure.bounds[1])
    ω = fftfreq(structure.ngrid, f)

    for i in @comps
        l = 1
        for j in @chain(i)
            for k in @chain(i)[l:end]
                R = (species.size[j] + species.size[k])*π
                Ω = 2*R .* (ω .== 0.0) + 2*sin.(ω.*R)./ω .*(ω .!= 0.0)
                Ω ./= 2
                map[:,j,k] = Ω./(R)
                map[:,k,j] = Ω./(R)
            end
            l+=1
        end
    end
    return TangentHSPropagator(map)
end


function propagate(system::DFTSystem, propagate::TangentHSPropagator, δf_res, ρ)
    model = system.model
    structure = system.structure
    ngrid = structure.ngrid
    species = system.species
    nbeads = sum(system.species.nbeads)

    Gcα = ones(Float64, ngrid, nbeads, nbeads)
    Gp  = ones(Float64, ngrid, nbeads)

    map = propagate.map

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

                            Gcα[:,k,α] = real.(ifft(fft(_Gcα).*map[:,k,α]))
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

                        Gp[:,k] = real.(ifft(fft(_Gp).*map[:,k,l]))
                    end
                end
            end
        end
    end

    return Gcα, Gp
end

export converge!