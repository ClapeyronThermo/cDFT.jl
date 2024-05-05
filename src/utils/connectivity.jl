import GCIdentifier: get_expanded_groups, get_mol, get_atoms, __getbondlist, get_grouplist

function get_connectivity(model::EoSModel, name::String)
    smiles = search_chemical(name).smiles
    mol = get_mol(smiles)
    atoms = get_atoms(mol)
    bondlist = __getbondlist(mol)
    groups = get_grouplist(model)
    group_id, mapping = get_expanded_groups(mol, groups, atoms, bondlist, false)

    bond_mat = zeros(Int64, length(group_id), length(group_id))

    for i in 1:length(bondlist)
        bond = bondlist[i]
        group_1 = mapping[:,bond[1]].==1
        group_2 = mapping[:,bond[2]].==1
        if group_1 == group_2
            continue
        else
            bond_mat[group_1, group_2] .+= 1
            bond_mat[group_2, group_1] .+= 1
        end
    end

    group_names = String[]
    
    for i in group_id
        pair = groups[i]
        push!(group_names, GCIdentifier.name(pair))
    end
    return group_id, group_names, bond_mat
end

