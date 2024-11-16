function TangentHSPropagator(model::EoSModel,species::DFTSpecies,structure::DFTStructure)
    ngrid = structure.ngrid
    nbeads = sum(species.nbeads)
    nd = dimension(structure)
    Ω = zeros(Float64, ngrid..., nbeads, nbeads)
    ω = structure_ω(structure)

    for i in @comps
        l = 1
        for j in @chain(i)
            for k in @chain(i)[l:end]
                
                R = (species.size[j] + species.size[k])*π
                for kk in CartesianIndices(ngrid)
                    n = Tuple(kk)
                    ω̄ = norm(@view(ω[n...,:]))
                    Ω[n...,j,k] = (2*R .* (ω̄ .== 0.0) + 2*sin.(ω̄.*R)./ω̄ .*(ω̄ .!= 0.0))/R/2
                    Ω[n...,k,j] = (2*R .* (ω̄ .== 0.0) + 2*sin.(ω̄.*R)./ω̄ .*(ω̄ .!= 0.0))/R/2
                end
                # selectdim(selectdim(map,nd+1,j),nd+1,k) .= Ω./(R)
                # selectdim(selectdim(map,nd+1,k),nd+1,j) .= Ω./(R)
            end
            l+=1
        end
    end

    plan = plan_fft(selectdim(selectdim(Ω,nd+1,1),nd+1,1), 1:nd)
    iplan = inv(plan)

    return TangentHSPropagator(Ω,plan,iplan)
end


function propagate(system::DFTSystem, propagate::TangentHSPropagator, δf_res, ρ)
    nd = dimension(system)
    model = system.model
    structure = system.structure
    ngrid = structure.ngrid
    species = system.species
    nbeads = sum(system.species.nbeads)

    Gcα = ones(Float64, ngrid..., nbeads, nbeads)
    Gp  = ones(Float64, ngrid..., nbeads)

    map = propagate.map
    P = propagate.plan
    iP = propagate.iplan

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
                                _Gcα = exp.(-selectdim(δf_res,nd+1,α)) .+ 0im
                            else
                                _Gcα = dropdims(exp.(-selectdim(δf_res,nd+1,α)).*prod(selectdim(selectdim(Gcα,nd+1,α),nd+1,β),dims=(nd+1,nd+2)); dims=nd+1) .+ 0im
                            end

                            matmul!(_Gcα,P,_Gcα)
                            elmul!(_Gcα,_Gcα,selectdim(selectdim(map,nd+1,k),nd+1,α))
                            matmul!(_Gcα,iP,_Gcα)
                            selectdim(selectdim(Gcα,nd+1,k),nd+1,α) .= real.(_Gcα)
                            # selectdim(selectdim(Gcα,nd+1,k),nd+1,α) .= real.(ifft(fft(_Gcα).*map[:,k,α]))
                        end
                    end
                end
            end

            # Get I2
            for L in 1:n_levels
                i_group_level = i_groups[findall(levels[i_groups].==L)]
                for k in i_group_level
                    if k == i_root
                        selectdim(Gp,nd+1,k) .= 1.
                    else
                        l = findall(n_intergroups[k,:] .&& levels.==L-1)[1]

                        α = findall(n_intergroups[l,:] .&& levels.==L)
                        α = α[α.!=k]
                    
                        _Gp = dropdims(exp.(-selectdim(δf_res,nd+1,l)).*selectdim(Gp,nd+1,l).*prod(selectdim(selectdim(Gcα,nd+1,l),nd+1,α),dims=(nd+1,nd+2)); dims=nd+1) .+ 0im
                        # println(_Gp)
                        matmul!(_Gp,P,_Gp)
                        elmul!(_Gp,_Gp,selectdim(selectdim(map,nd+1,k),nd+1,l))
                        matmul!(_Gp,iP,_Gp)
                        selectdim(Gp,nd+1,k) .= real.(_Gp)

                        # ifft(fft(_Gp).*map[:,k,α])
                        # selectdim(Gp,nd+1,k) .= real.(ifft(fft(_Gp).*selectdim(selectdim(map,nd+1,k),nd+1,l)))
                    end
                end
            end
        end
    end

    return Gcα, Gp
end

export converge!