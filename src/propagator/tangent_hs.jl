function TangentHSPropagator(model::EoSModel,species::DFTSpecies,structure::DFTStructure,device::Backend)
    ngrid = structure.ngrid
    nbeads = sum(species.nbeads)
    nd = dimension(structure)
    Ω = allocate(device,ComplexF64,ngrid...,nbeads,nbeads)
    ω = structure_ω(structure, device)
    ω = Adapt.adapt(device, ω)
    for i in @comps
        l = 1
        for j in @chain(i)
            for k in @chain(i)[l:end]
                R = (species.size[j] + species.size[k])*π
                
                ω̄ = dropdims(sqrt.(sum(abs2, ω, dims=nd+1)), dims=nd+1)  # lives on same backend as ω


                val = @. (2*R * (ω̄ == 0.0) + 2*sin(ω̄*R)/max(ω̄, eps()) * (ω̄ != 0.0)) / R / 2
                
                selectdim(selectdim(Ω, nd+1, j), nd+1, k) .= val
                selectdim(selectdim(Ω, nd+1, k), nd+1, j) .= val
            end
            l += 1
        end
    end

    return TangentHSPropagator(Ω)
end

function preallocate_propagator(system::AbstractcDFTSystem,propagator::TangentHSPropagator,ρ,backend::Backend)
    nd = dimension(system)
    ngrid = system.structure.ngrid
    Gcα = allocate(backend, Float64, size(ρ)..., sum(system.species.nbeads))
    Gcα .= 1.0
    Gp = allocate(backend, Float64, size(ρ)...)
    Gp .= 1.0
    buf = similar(selectdim(ρ,nd+1,1), ComplexF64)
    scratch = allocate(backend, Float64, ngrid...)

    if backend isa CPU
        plan = plan_fft!(buf, 1:length(ngrid); num_threads=Threads.nthreads())
    else
        plan = plan_fft!(buf, 1:length(ngrid))
    end
    return Gcα, Gp, buf, plan, inv(plan), scratch
end


function propagate!(system::AbstractcDFTSystem, propagate::TangentHSPropagator, ρ, δfδρ_res, Gcα, Gp, buf, P, iP, scratch)
    nd = dimension(system)
    model = system.model
    structure = system.structure
    ngrid = structure.ngrid
    species = system.species
    nbeads = sum(system.species.nbeads)

    map = propagate.map

    levels = species.levels
    for i in @comps
        if system.species.nbeads[i] !== 1
            n_intergroups = model.groups.n_intergroups[i] .== 1
            i_groups = model.groups.i_groups[i]
            n_levels = maximum(levels[i_groups])

            i_root = i_groups[levels[i_groups].==1][1]
            is_leaf = sum(n_intergroups,dims=1).==1 .&& (levels.!=1)'
            # Bottom-up pass: compute Gcα
            for L in n_levels:-1:1
                i_group_level = i_groups[findall(levels[i_groups].==L)]
                for k in i_group_level
                    k_children = findall(n_intergroups[k,:] .&& levels.==L+1)
                    if !is_leaf[k]
                        for α in k_children
                            β = findall(n_intergroups[α,:] .&& levels.==L+2)
                            buf .= exp.(-selectdim(δfδρ_res, nd+1, α)) .+ 0im
                            for β_k in β
                                buf .*= selectdim(selectdim(Gcα, nd+1, α), nd+1, β_k)
                            end
                            convolve!(selectdim(selectdim(Gcα,nd+1,k),nd+1,α), buf, selectdim(selectdim(map,nd+1,k),nd+1,α), P, iP, buf)
                        end
                    end
                end
            end

            # Top-down pass: compute Gp
            for L in 1:n_levels
                i_group_level = i_groups[findall(levels[i_groups].==L)]
                for k in i_group_level
                    if k == i_root
                        selectdim(Gp,nd+1,k) .= 1.
                    else
                        l = findall(n_intergroups[k,:] .&& levels.==L-1)[1]

                        α = findall(n_intergroups[l,:] .&& levels.==L)
                        α = α[α.!=k]

                        buf .= exp.(-selectdim(δfδρ_res, nd+1, l)) .* selectdim(Gp, nd+1, l) .+ 0im
                        for α_k in α
                            buf .*= selectdim(selectdim(Gcα, nd+1, l), nd+1, α_k)
                        end
                        convolve!(selectdim(Gp,nd+1,k), buf, selectdim(selectdim(map,nd+1,l),nd+1,k), P, iP, buf)
                    end
                end
            end
        end
    end

    # Final update: subtract chain-bonding contributions from δfδρ_res
    for i in @comps
        for j in @chain(i)
            if system.species.nbeads[i] != 1
                α_vec = findall(model.groups.n_intergroups[i][j,:] .== 1 .&& species.levels .> species.levels[j])
            else
                α_vec = (j,)
            end
            scratch .= log.(selectdim(Gp, nd+1, j))
            for α_k in α_vec
                scratch .+= log.(selectdim(selectdim(Gcα, nd+2, α_k), nd+1, j))
            end
            selectdim(δfδρ_res, nd+1, j) .-= scratch
        end
    end
end