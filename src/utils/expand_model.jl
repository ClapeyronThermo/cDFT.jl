function expand_model(model::EoSModel)
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

    n_groups_cache = Clapeyron.pack_vectors([Float64.(n_flattenedgroups[i]) for i in 1:nspecies])

    if typeof(model.groups) <: Clapeyron.StructGroupParam
        n_intergroups = Matrix{Int64}[]
        for i in 1:nspecies
            _n_intergroups = zeros(Int64,ngroups,ngroups)
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
        end
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
    else
        groupsparams = GroupParam(model.components,
                                groups,
                                model.groups.grouptype,
                                n_groups,
                                i_groups,
                                flattenedgroups,
                                n_flattenedgroups,
                                n_groups_cache,
                                model.groups.sourcecsvs)
    end


    # Expand the sites 
    assoc_groups = split.(model.sites.flattenedsites,"/")
    assoc_sites = [assoc_groups[i][2] for i in 1:length(assoc_groups)]
    assoc_groups = [assoc_groups[i][1] for i in 1:length(assoc_groups)]
    n_sites_per_group = sum(model.sites.n_flattenedsites)

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
        _n_flattenedsites = zeros(Int64,ngroups)
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

                values = Clapeyron.Compressed4DMatrix(values,outer_indices,inner_indices)
                append!(params, [AssocParam(name,
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

    eos_type = typeof(model)

    return new_model = eos_type(model.components,
                                groupsparams,
                                siteparams,
                                eosparams,
                                model.idealmodel,
                                model.assoc_options,
                                model.references)
end