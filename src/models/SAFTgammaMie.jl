using Clapeyron: SAFTgammaMieModel
using Clapeyron: d_gc_av

function DFTSystem(model::SAFTgammaMieModel,structure::DFTStructure,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure)
    propagator = get_propagator(model, species, structure)
    return DFTSystem(model, species, structure, fields, propagator, options)
end

struct SAFTgammaMieSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::SAFTgammaMie,structure::DFTStructure)
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk 
    HSd = d(model,1e-3,T,ones(length(model.groups.flattenedgroups)))

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk./sum(ρbulk)) / Clapeyron.R̄ / T
    nbeads = length.(model.groups.groups)

    levels = zeros(Int, sum(nbeads))

    for i in @comps
        i_groups = model.groups.i_groups[i]
        bond_mat = Bool.(model.groups.n_intergroups[i])
        nbonds = sum(bond_mat,dims=2)[:]
        is_leaf = nbonds .== 1
        i_root = i_groups[findfirst(nbonds[i_groups] .== maximum(nbonds[i_groups]))]
        levels[i_root] = 1
    
        idx_current_level = i_root
        is_bonded = bond_mat[idx_current_level,:]
        k = 1
        while any(levels[i_groups] .== 0)
            levels[is_bonded] .= k+1
            idx_next_level = findall(levels .== k+1 .&& .!(is_leaf))
            is_bonded = (sum(bond_mat[idx_next_level,:],dims=1)[:].==1 .&& levels.==0)
            k+=1
        end
    end
    return SAFTgammaMieSpecies(nbeads,HSd,levels,ρbulk,μres)
end

function get_fields(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure)
    nb = sum(species.nbeads)
    f = structure.ngrid/(structure.bounds[2]-structure.bounds[1])
    ω = fftfreq(structure.ngrid, f)
    d = species.size
    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = d ./ σ
    ψ = @. cbrt(3*C*(1/(λ_a-3)-1/(λ_r-3)))
    return [WeightedDensity(:ρ,zeros(nb),ω),
            WeightedDensity(:∫ρdz,0.5*d,ω),
            WeightedDensity(:∫ρz²dz,0.5*d,ω),
            WeightedDensity(:∫ρzdz,0.5*d,ω),
            WeightedDensity(:∫ρz²dz,d,ω),
            WeightedDensity(:∫ρdz,d,ω),
            WeightedDensity(:∫ρz²dz,ψ.*d,ω)]
end


function get_propagator(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure)
    return TangentHSPropagator(model, species, structure)
end

function expand_model(model::SAFTgammaMieModel)
    nspecies = length(model)

    # Expand the groups
    ngroup_types = length(model.groups.flattenedgroups)
    ngroups = sum(sum.(model.groups.n_flattenedgroups))
    
    ngroups_k = zeros(Int64,ngroup_types)
    n_groups = Vector{Int64}[]
    for i in 1:nspecies
        ngroups_k .+= model.groups.n_flattenedgroups[i]    
        append!(n_groups, [ones(Int64,sum(model.groups.n_groups[i]))])
    end

    flattenedgroups = String[]
    n_flattenedgroups = [zeros(Int64,ngroups) for i in 1:nspecies]

    for i in 1:ngroup_types
        append!(flattenedgroups, model.groups.flattenedgroups[i]*"_".*string.(Int.(1:ngroups_k[i])))
        k=sum(ngroups_k[1:i-1])
        for j in 1:nspecies
            n_flattenedgroups[j][k+1:k+model.groups.n_flattenedgroups[j][i]] .= 1
            k+=model.groups.n_flattenedgroups[j][i]
        end
    end

    groups = []
    i_groups = []
    for i in 1:nspecies
        append!(groups, [flattenedgroups[n_flattenedgroups[i].==1]])
        append!(i_groups,[deleteat!(n_flattenedgroups[i].*(1:ngroups),findall(n_flattenedgroups[i].==0))])
    end

    n_intergroups = Matrix{Int64}[]
    for i in 1:nspecies
        _n_intergroups = zeros(Int64,ngroups,ngroups)
        if length(n_groups[i]) > 1
            _, group_names, bondmat = cDFT.get_connectivity(model,model.components[i])
            _groups = getindex.(split.(groups[i],"_"),1)
            ngroup_types_i = length(model.groups.groups[i])
            for k in 1:ngroup_types_i
                idx_group_i_1 = findall(_groups.==model.groups.groups[i][k])
                idx_group_1_2 = findall(group_names.==model.groups.groups[i][k])
                for l in 1:ngroup_types_i
                    idx_group_j_1 = findall(_groups.==model.groups.groups[i][l])
                    idx_group_j_2 = findall(group_names.==model.groups.groups[i][l])
                    _n_intergroups[i_groups[i][idx_group_i_1],i_groups[i][idx_group_j_1]] = bondmat[idx_group_1_2,idx_group_j_2]
                end
            end
            append!(n_intergroups, [_n_intergroups])
        else
            append!(n_intergroups, [_n_intergroups])
        end
    end
    


    # Expand the sites 
    assoc_groups = split.(model.vrmodel.sites.flattenedsites,"/")
    assoc_sites = [assoc_groups[i][2] for i in 1:length(assoc_groups)]
    assoc_groups = [assoc_groups[i][1] for i in 1:length(assoc_groups)]

    n_sites_per_group = sum(model.vrmodel.sites.n_flattenedsites)

    flattenedsites = String[]

    n_sites_per_group_expanded = Int64[]
    
    for i in 1:length(assoc_groups)
        group_idx = findfirst(model.groups.flattenedgroups.==assoc_groups[i])
        n_sites_per_group[i] /= ngroups_k[group_idx]
        append!(flattenedsites, model.groups.flattenedgroups[group_idx]*"_".*string.(1:ngroups_k[group_idx]).*"/".*assoc_sites[i])
        append!(n_sites_per_group_expanded, n_sites_per_group[i]*ones(Int64,ngroups_k[group_idx]))
    end

    nsites = length(flattenedsites)
    assoc_sites = unique(assoc_sites)

    sites = []
    i_sites = Vector{Int64}[]
    i_flattenedsites = Vector{Int64}[]
    n_sites = Vector{Int64}[]  
    n_flattenedsites = Vector{Int64}[]  
    site_translator = Vector{Tuple{Int64,Int64}}[]
    k = 1
    for i in 1:nspecies
        sites_per_species = String[]
        _i_flattenedsites = zeros(Int64,nsites)
        _i_sites = Int64[]
        _n_flattenedsites = zeros(Int64,nsites)
        _n_sites = Int64[]
        _site_translator = Tuple{Int64,Int64}[]
        l = 1
        for j in 1:nsites
            group_name = split(flattenedsites[j],"/")[1]
            site_name = split(flattenedsites[j],"/")[2]
            if group_name in groups[i]
                append!(sites_per_species, [flattenedsites[j]])
                append!(_i_sites, [j])
                append!(_n_sites, [n_sites_per_group_expanded[j]])
                group_idx = findfirst(flattenedgroups.==group_name)
                site_idx = findfirst(assoc_sites.==site_name)
                append!(_site_translator,[(group_idx,site_idx)])
                _i_flattenedsites[j] = l
                _n_flattenedsites[j] = n_sites_per_group_expanded[j]
                l+=1
            end
        end
        # _i_flattenedsites[k:k+length(sites_per_species)-1] = 1:length(sites_per_species)
        append!(sites, [sites_per_species])
        append!(i_sites,[_i_sites])
        append!(i_flattenedsites,[_i_flattenedsites])
        append!(n_sites,[_n_sites])
        append!(n_flattenedsites,[_n_flattenedsites])
        append!(site_translator,[_site_translator])
        k += length(sites_per_species)
    end

    siteparams = SiteParam(model.components,
                           sites,
                           Clapeyron.pack_vectors(n_sites),
                           i_sites,
                           flattenedsites,
                           n_flattenedsites,
                           i_flattenedsites,
                           model.sites.sourcecsvs,
                           site_translator)

    # Expand the parameters
    params_names = fieldnames(typeof(model.params))
    nparams = length(params_names)

    params = []

    for i in 1:nparams
        param = getfield(model.params,params_names[i])
        name = param.name
        if typeof(param) <: SingleParam
            if param.components == model.components
                append!(params,[param])
            else
                values = zeros(Float64,ngroups)
                ismissingvalues = zeros(Bool,ngroups)
                k = 1
                for j in 1:ngroup_types
                    values[k:k-1+ngroups_k[j]] .= param.values[j]
                    ismissingvalues[k:k-1+ngroups_k[j]] .= param.ismissingvalues[j]
                    k+=ngroups_k[j]
                end
                append!(params, [SingleParam(name,
                                            flattenedgroups,
                                            values,
                                            ismissingvalues,
                                            param.sourcecsvs,
                                            param.sources)])
            end
        elseif typeof(param) <: PairParam
            if param.components == model.components
                append!(params,[param])
            else
                values = zeros(Float64,ngroups,ngroups)
                ismissingvalues = zeros(Bool,ngroups,ngroups)
                k = 1
                for a in 1:ngroup_types
                    l = sum(ngroups_k[1:a-1])+1
                    for b in a:ngroup_types
                        values[k:k-1+ngroups_k[a],l:l-1+ngroups_k[b]] .= param.values[a,b]
                        ismissingvalues[k:k-1+ngroups_k[a],l:l-1+ngroups_k[b]] .= param.ismissingvalues[a,b]
                        values[l:l-1+ngroups_k[b],k:k-1+ngroups_k[a]] .= param.values[a,b]
                        ismissingvalues[l:l-1+ngroups_k[b],k:k-1+ngroups_k[a]] .= param.ismissingvalues[a,b]
                        l+=ngroups_k[b]
                    end
                    k+=ngroups_k[a]
                end
                append!(params, [PairParam(name,
                                            flattenedgroups,
                                            values,
                                            ismissingvalues,
                                            param.sourcecsvs,
                                            param.sources)])
            end
        elseif typeof(param) <: AssocParam
            param = getfield(model.vrmodel.params,params_names[i])
            if length(param.values.values) < 1
                append!(params, [param])
            else
                values = Float64[]
                inner_indices = Tuple{Int64,Int64}[]
                outer_indices = Tuple{Int64,Int64}[]
                n_interaction = length(param.values.values)
                assoc_groups = Vector{String}[]
                assoc_sites  = Vector{String}[]
                for i in 1:nspecies
                    append!(assoc_groups,[getindex.(split.(getindex.(split.(sites[i],"/"),1),"_"),1)])
                    append!(assoc_sites, [getindex.(split.(sites[i],"/"),2)])
                end

                for i in 1:n_interaction
                    value = param.values.values[i]
                    id_species_1, id_species_2 = param.values.outer_indices[i]
                    id_site_1, id_site_2 = param.values.inner_indices[i]
                    group_type_1,site_type_1 = split(param.sites[id_species_1][id_site_1],"/")
                    group_type_2,site_type_2 = split(param.sites[id_species_2][id_site_2],"/")

                    inner_idx_1 = findall(assoc_groups[id_species_1].==group_type_1 .&& assoc_sites[id_species_1].==site_type_1)
                    inner_idx_2 = findall(assoc_groups[id_species_2].==group_type_2 .&& assoc_sites[id_species_2].==site_type_2)

                    for i in 1:length(inner_idx_1)
                        for j in 1:length(inner_idx_2)
                            append!(values,[value])
                            append!(inner_indices,[(inner_idx_1[i],inner_idx_2[j])])
                            append!(outer_indices,[(id_species_1,id_species_2)])
                        end
                    end
                end

                types = eltype(values)
                values = Clapeyron.Compressed4DMatrix(values,outer_indices,inner_indices)
                # println(components)
                append!(params, [AssocParam{types}(name,
                                            model.components,
                                            values,
                                            sites,
                                            param.sourcecsvs,
                                            param.sources)])
            end

        end
    end
    eosparam_type = typeof(model.params)
    eosparams = eosparam_type(params...)

    n_groups_cache = Clapeyron.pack_vectors([Float64.(n_flattenedgroups[i].*eosparams.shapefactor.values.*eosparams.segment.values) for i in 1:nspecies])

    groupsparams = Clapeyron.StructGroupParam(model.components,
                                    groups,
                                    model.groups.grouptype,
                                    n_groups,
                                    n_intergroups,
                                    i_groups,
                                    flattenedgroups,
                                    n_flattenedgroups,
                                    n_groups_cache,
                                    model.groups.sourcecsvs)


    eos_type = typeof(model)

    vreosparam_type = typeof(model.vrmodel.params)
    vreosparams = vreosparam_type(model.vrmodel.params.Mw,
                                  model.vrmodel.params.segment,
                                  model.vrmodel.params.sigma,
                                  model.vrmodel.params.lambda_a,
                                  model.vrmodel.params.lambda_r,
                                  model.vrmodel.params.epsilon,
                                  eosparams.epsilon_assoc,
                                  eosparams.bondvol)

    vrmodel = SAFTVRMie(model.vrmodel.components,
                        siteparams,
                        vreosparams,
                        model.vrmodel.idealmodel,
                        model.vrmodel.assoc_options,
                        model.vrmodel.references
                        )

    return new_model = eos_type(model.components,
                                groupsparams,
                                siteparams,
                                eosparams,
                                model.idealmodel,
                                vrmodel,
                                model.epsilon_mixing,
                                model.assoc_options,
                                model.references)
end

function f_res(system::DFTSystem, model::SAFTgammaMieModel, n)
    n1,n2,n3,n4,n5,n6,n7 = @view(n[1,:]),@view(n[2,:]),@view(n[3,:]),@view(n[4,:]),@view(n[5,:]),@view(n[6,:]),@view(n[7,:])

    return f_hs(system,model,n2,n3,n4) + f_disp(system,model,n7) + f_chain(system,model,n1,n5,n6) + f_assoc(system,model,n2,n3,n4)
end

function f_hs(system::DFTSystem, model::SAFTgammaMieModel, n, n₃, nᵥ)
    species = system.species
    m = model.params.segment.values
    S = model.params.shapefactor.values
    HSd = species.size

    n₀ = zero(first(n) + first(m) + first(HSd))
    n₁,n₂,nᵥ₁,nᵥ₂,n₃₃ = zero(n₀), zero(n₀), zero(n₀), zero(n₀), zero(n₀)
    for i in 1:length(n)
        mᵢ,Sᵢ,HSdᵢ,nᵥᵢ = m[i],S[i],HSd[i],nᵥ[i]
        nᵢmᵢ = n[i]*mᵢ*Sᵢ
        n₀ += nᵢmᵢ/HSdᵢ
        n₁ += 0.5nᵢmᵢ
        n₂ += π*nᵢmᵢ*HSdᵢ
        nᵥ₁ += nᵥᵢ*mᵢ*Sᵢ/HSdᵢ
        nᵥ₂ += -2π*nᵥᵢ*mᵢ*Sᵢ
        n₃₃ += n₃[i]*mᵢ*Sᵢ
    end

    return -n₀*log(1-n₃₃)+(n₁*n₂-nᵥ₂*nᵥ₁)/(1-n₃₃)+(n₂^3/3-n₂*nᵥ₂*nᵥ₂)*(log(1-n₃₃)/(12*π*n₃₃^2)+1/(12*π*n₃₃*(1-n₃₃)^2))
end

function f_disp(system::DFTSystem, model::SAFTgammaMieModel, ρ̄)
    V = nothing
    ψ = system.fields[end].width
    _d = system.species.size
    T = system.structure.conditions[2]
    m = model.params.segment.values
    S = model.params.shapefactor.values
    _ϵ = model.params.epsilon
    _λr = model.params.lambda_r
    _λa = model.params.lambda_a
    _σ = model.params.sigma

    m = m.*S

    ρ̄ = ρ̄*3 ./(4*ψ.^3)/π
    ∑ρ̄ = sum(ρ̄)
    z = ρ̄ /∑ρ̄
    m̄ = dot(z,m)
    m̄inv = 1/m̄
    ∑z = sum(z)

    ρS = dot(ρ̄,m)

    _ζ_X = zero(T+first(ρ̄)+one(eltype(model)))
    kρS = ρS* π/6/8
    σ3_x = _ζ_X

    for i ∈ @groups
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        di =_d[i]
        r1 = kρS*x_Si*x_Si*(2*di)^3
        _ζ_X += r1
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3_x += 2*x_Si*x_Sj*(_σ[i,j]^3)
            dij = (di + _d[j])
            r1 = kρS*x_Si*x_Sj*dij^3
            _ζ_X += 2*r1
        end
    end

    _ζst = σ3_x*ρS*π/6
    
    a₁ = zero(T+first(z)+one(eltype(model)))
    a₂ = a₁
    a₃ = a₁
    _ζst5 = _ζst^5
    _ζst8 = _ζst^8
    _KHS = @f(KHS,_ζ_X,ρS)
    for i ∈ @groups
        j = i
        x_Si = z[i]*m[i]*m̄inv
        x_Sj = x_Si
        ϵ = _ϵ[i,j]
        λa = _λa[i,i]
        λr = _λr[i,i]
        σ = _σ[i,i]
        _C = @f(Cλ,λa,λr)
        dij = _d[i]
        dij3 = dij^3
        x_0ij = σ/dij
        #calculations for a1 - diagonal
        aS_1_a = @f(aS_1,λa,_ζ_X)
        aS_1_r = @f(aS_1,λr,_ζ_X)
        B_a = @f(B,λa,x_0ij,_ζ_X)
        B_r = @f(B,λr,x_0ij,_ζ_X)
        a1_ij = (2*π*ϵ*dij3)*_C*ρS*
        (x_0ij^λa*(aS_1_a+B_a) - x_0ij^λr*(aS_1_r+B_r))

        #calculations for a2 - diagonal
        aS_1_2a = @f(aS_1,2*λa,_ζ_X)
        aS_1_2r = @f(aS_1,2*λr,_ζ_X)
        aS_1_ar = @f(aS_1,λa+λr,_ζ_X)
        B_2a = @f(B,2*λa,x_0ij,_ζ_X)
        B_2r = @f(B,2*λr,x_0ij,_ζ_X)
        B_ar = @f(B,λr+λa,x_0ij,_ζ_X)
        α = _C*(1/(λa-3)-1/(λr-3))
        f1,f2,f3,f4,f5,f6 = @f(f123456,α)
        _χ = f1*_ζst+f2*_ζst5+f3*_ζst8
        a2_ij = π*_KHS*(1+_χ)*ρS*ϵ^2*dij3*_C^2 *
        (x_0ij^(2*λa)*(aS_1_2a+B_2a)
        - 2*x_0ij^(λa+λr)*(aS_1_ar+B_ar)
        + x_0ij^(2*λr)*(aS_1_2r+B_2r))

        #calculations for a3 - diagonal
        a3_ij = -ϵ^3*f4*_ζst * exp(f5*_ζst+f6*_ζst^2)
        #adding - diagonal
        a₁ += a1_ij*x_Si*x_Si
        a₂ += a2_ij*x_Si*x_Si
        a₃ += a3_ij*x_Si*x_Si
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            ϵ = _ϵ[i,j]
            λa = _λa[i,j]
            λr = _λr[i,j]
            σ = _σ[i,j]
            _C = @f(Cλ,λa,λr)
            dij = 0.5*(_d[i]+_d[j])
            x_0ij = σ/dij
            dij3 = dij^3
            x_0ij = σ/dij
            #calculations for a1
            a1_ij = (2*π*ϵ*dij3)*_C*ρS*
            (x_0ij^λa*(@f(aS_1,λa,_ζ_X)+@f(B,λa,x_0ij,_ζ_X)) - x_0ij^λr*(@f(aS_1,λr,_ζ_X)+@f(B,λr,x_0ij,_ζ_X)))

            #calculations for a2
            α = _C*(1/(λa-3)-1/(λr-3))
            f1,f2,f3,f4,f5,f6 = @f(f123456,α)
            _χ = f1*_ζst+f2*_ζst5+f3*_ζst8
            a2_ij = π*_KHS*(1+_χ)*ρS*ϵ^2*dij3*_C^2 *
            (x_0ij^(2*λa)*(@f(aS_1,2*λa,_ζ_X)+@f(B,2*λa,x_0ij,_ζ_X))
            - 2*x_0ij^(λa+λr)*(@f(aS_1,λa+λr,_ζ_X)+@f(B,λa+λr,x_0ij,_ζ_X))
            + x_0ij^(2*λr)*(@f(aS_1,2λr,_ζ_X)+@f(B,2*λr,x_0ij,_ζ_X)))

            #calculations for a3
            a3_ij = -ϵ^3*f4*_ζst * exp(f5*_ζst+f6*_ζst^2)
            #adding
            a₁ += 2*a1_ij*x_Si*x_Sj
            a₂ += 2*a2_ij*x_Si*x_Sj
            a₃ += 2*a3_ij*x_Si*x_Sj
        end
    end
    a₁ = a₁*m̄/T/∑z #/sum(z)
    a₂ = a₂*m̄/(T*T)/∑z  #/sum(z)
    a₃ = a₃*m̄/(T*T*T)/∑z  #/sum(z)
    #@show (a₁,a₂,a₃)
    adisp = a₁ + a₂ + a₃
    return ∑ρ̄*adisp
end

function f_chain(system::DFTSystem, model::SAFTgammaMieModel, ρhc, ρ̄hc, _λ)
    V = nothing
    T = system.structure.conditions[2]
    x = system.structure.ρbulk / sum(system.structure.ρbulk)
    
    m = model.vrmodel.params.segment
    m_gc = model.params.segment.values .* model.params.shapefactor.values
    _ϵ = model.vrmodel.params.epsilon
    _λr = model.vrmodel.params.lambda_r
    _λa = model.vrmodel.params.lambda_a
    _σ = model.vrmodel.params.sigma
    _σ_gc = model.params.sigma.values
    _d = d_gc_av(model,V,T,x,system.species.size)

    ρ̄hc = ρ̄hc*3 ./(4 .*system.species.size.^3)/π
    _λ =_λ ./ (2*system.species.size)

    _ρhc = zeros(eltype(ρhc),length(model))
    _ρ̄hc = zeros(eltype(ρ̄hc),length(model))
    λ = zeros(eltype(_λ),length(model))

    for i in @comps
        for k in @groups(i)
            _ρhc[i] += ρhc[k]/system.species.nbeads[i]
            _ρ̄hc[i] += ρ̄hc[k]/system.species.nbeads[i]
            λ[i] += _λ[k]/system.species.nbeads[i]
        end
    end

    z = _ρ̄hc /sum(_ρ̄hc)
    z_gc = ρhc /sum(ρhc)

    m̄ = dot(z,m)
    m̄_gc = dot(z_gc,m_gc)
    m̄inv_gc = 1/m̄_gc

    ρS = dot(_ρ̄hc,m)

    _ζ_X = zero(T+first(_ρ̄hc)+one(eltype(model)))
    kρS = ρS* π/6/8
    σ3_x = deepcopy(_ζ_X)

    for i ∈ @groups
        x_Si = z_gc[i]*m_gc[i]*m̄inv_gc
        σ3_x += x_Si*x_Si*(_σ_gc[i,i]^3)
        di =system.species.size[i]
        r1 = kρS*x_Si*x_Si*(2*di)^3
        _ζ_X += r1
        for j ∈ 1:(i-1)
            x_Sj = z_gc[j]*m_gc[j]*m̄inv_gc
            σ3_x += 2*x_Si*x_Sj*(_σ_gc[i,j]^3)
            dij = (di + system.species.size[j])
            r1 = kρS*x_Si*x_Sj*dij^3
            _ζ_X += 2*r1
        end
    end

    _ζst = σ3_x*ρS*π/6

    fchain = zero(T+first(z)+one(eltype(model)))
    _KHS,_∂KHS = @f(KHS_fdf,_ζ_X,ρS)
    for i ∈ @comps
        ϵ = _ϵ[i,i]
        λa = _λa[i,i]
        λr = _λr[i,i]
        σ = _σ[i,i]
        _C = @f(Cλ,λa,λr)
        dij = _d[i]
        x_0ij = σ/dij
        x_0ij = σ/dij
        #calculations for a1 - diagonal
        aS_1_a,∂aS_1∂ρS_a = @f(aS_1_fdf,λa,_ζ_X,ρS)
        aS_1_r,∂aS_1∂ρS_r = @f(aS_1_fdf,λr,_ζ_X,ρS)
        B_a,∂B∂ρS_a = @f(B_fdf,λa,x_0ij,_ζ_X,ρS)
        B_r,∂B∂ρS_r = @f(B_fdf,λr,x_0ij,_ζ_X,ρS)

        #calculations for a2 - diagonal
        aS_1_2a,∂aS_1∂ρS_2a = @f(aS_1_fdf,2*λa,_ζ_X,ρS)
        aS_1_2r,∂aS_1∂ρS_2r = @f(aS_1_fdf,2*λr,_ζ_X,ρS)
        aS_1_ar,∂aS_1∂ρS_ar = @f(aS_1_fdf,λa+λr,_ζ_X,ρS)
        B_2a,∂B∂ρS_2a = @f(B_fdf,2*λa,x_0ij,_ζ_X,ρS)
        B_2r,∂B∂ρS_2r = @f(B_fdf,2*λr,x_0ij,_ζ_X,ρS)
        B_ar,∂B∂ρS_ar = @f(B_fdf,λr+λa,x_0ij,_ζ_X,ρS)
        α = _C*(1/(λa-3)-1/(λr-3))
        g_HSi = @f(g_HS,x_0ij,_ζ_X)
        #@show (g_HSi,i)
        ∂a_1∂ρ_S = _C*(x_0ij^λa*(∂aS_1∂ρS_a+∂B∂ρS_a)
                      - x_0ij^λr*(∂aS_1∂ρS_r+∂B∂ρS_r))
        #@show (∂a_1∂ρ_S,1)

        g_1_ = 3*∂a_1∂ρ_S-_C*(λa*x_0ij^λa*(aS_1_a+B_a)-λr*x_0ij^λr*(aS_1_r+B_r))
        #@show (g_1_,i)
        θ = exp(ϵ/T)-1
        γc = 10 * (-tanh(10*(0.57-α))+1) * _ζst*θ*exp(-6.7*_ζst-8*_ζst^2)
        ∂a_2∂ρ_S = 0.5*_C^2 *
            (ρS*_∂KHS*(x_0ij^(2*λa)*(aS_1_2a+B_2a)
            - 2*x_0ij^(λa+λr)*(aS_1_ar+B_ar)
            + x_0ij^(2*λr)*(aS_1_2r+B_2r))
            + _KHS*(x_0ij^(2*λa)*(∂aS_1∂ρS_2a+∂B∂ρS_2a)
            - 2*x_0ij^(λa+λr)*(∂aS_1∂ρS_ar+∂B∂ρS_ar)
            + x_0ij^(2*λr)*(∂aS_1∂ρS_2r+∂B∂ρS_2r)))

        gMCA2 = 3*∂a_2∂ρ_S-_KHS*_C^2 *
        (λr*x_0ij^(2*λr)*(aS_1_2r+B_2r)-
            (λa+λr)*x_0ij^(λa+λr)*(aS_1_ar+B_ar)+
            λa*x_0ij^(2*λa)*(aS_1_2a+B_2a))
        g_2_ = (1+γc)*gMCA2
        #@show (g_2_,i)
        g_Mie_ = g_HSi*exp(ϵ/T*g_1_/g_HSi+(ϵ/T)^2*g_2_/g_HSi)
        #@show (g_Mie_,i)
        fchain +=  _ρhc[i]*(log(g_Mie_)*(m[i]-1))
    end
    
    return -fchain
end

function Δ(model::SAFTgammaMieModel, T, n, n₃, nᵥ, i, j, a, b)
    _d = d(model,1e-3,T,ones(length(model.groups.flattenedgroups)))
    _σ = model.params.sigma.values
    m = model.params.segment.values
    S = model.params.shapefactor.values
    ϵ_assoc = model.params.epsilon_assoc.values
    K = model.params.bondvol.values[i,j][a,b]
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(K))
    iszero(K) && return _0

    ρ̄ = n₃*3*2 ./(_d.^3)/π
    m = m.*S
    z = ρ̄ /sum(ρ̄)
    m̄ = dot(z,m)
    m̄inv = 1/m̄

    ρS = dot(ρ̄,m)

    σ3_x = zero(T+first(z)+one(eltype(model)))

    for i ∈ @groups
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3_x += 2*x_Si*x_Sj*(_σ[i,j]^3)
        end
    end
    ρr  = ρS*σ3_x
    
    ϵ = model.vrmodel.params.epsilon
    Tr = T/ϵ[i,j]
    _I = I(model,Tr,ρr)
    
    F = expm1(ϵ_assoc[i,j][a,b]/T)

    return F*K*_I
end

function I(model::SAFTgammaMieModel, Tr,ρr)
    c  = SAFTVRMieconsts.c
    res = zero(ρr+Tr)
    @inbounds for n ∈ 0:10
        ρrn = ρr^n
        res_m = zero(res)
        for m ∈ 0:(10-n)
            res_m += c[n+1,m+1]*Tr^m
        end
        res += res_m*ρrn
    end
    return res
end