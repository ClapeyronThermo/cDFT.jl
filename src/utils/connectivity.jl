abstract type MolStructure end

struct SMILESStructure <: MolStructure
    smiles::String
end
smiles(s::String) = SMILESStructure(s)

struct CustomStructure <: MolStructure
    string::String
end
custom_structure(s::String) = CustomStructure(s)

# Parser for single-letter group notation with SMILES-style branching.
# Each non-paren character = one group instance; parentheses open/close branches.
# Example: "AAAAABBBB" -> linear A₁-...-A₅-B₁-...-B₄
#          "A(B)CCC"   -> B branches off A, then C-C-C continues from A
function get_connectivity(::EoSModel, cs::CustomStructure)
    s = cs.string
    group_names = Char[]
    bonds = Pair{Int,Int}[]
    stack = Int[]
    current = 0

    for ch in s
        if ch == '('
            push!(stack, current)
        elseif ch == ')'
            current = pop!(stack)
        else
            push!(group_names, ch)
            idx = length(group_names)
            if current != 0
                push!(bonds, current => idx)
            end
            current = idx
        end
    end

    n = length(group_names)
    bond_mat = zeros(Int64, n, n)
    for (i, j) in bonds
        bond_mat[i, j] = 1
        bond_mat[j, i] = 1
    end

    names = string.(group_names)
    return 1:n, names, bond_mat
end

# Fallback when GCIdentifier/ChemicalIdentifiers extension is not loaded
function get_connectivity(::EoSModel, name::String)
    error("""
    Auto-detection of GC connectivity from a chemical name requires GCIdentifier and
    ChemicalIdentifiers. Either:
      • load both packages before constructing the DFTSystem, or
      • supply connectivity manually via `mol_structure`:
          DFTSystem(model, structure, options;
              mol_structure = Dict("$(name)" => custom_structure("AAAAABBBB")))
          DFTSystem(model, structure, options;
              mol_structure = Dict("$(name)" => smiles("CCCCCC")))
    """)
end
