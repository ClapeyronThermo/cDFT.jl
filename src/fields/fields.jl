abstract type ScalarField <: DFTField end
abstract type VectorField <: DFTField end

include("weighted_densities.jl")

"""
    evaluate_field(system::DFTSystem, profiles)

This function will obtain every field used in the system (listed in `system.fields`). The output will be a 3D array with the dimensions `(ngrid,nf,nc)`, where `ngrid` is the number of grid points, `nf` is the number of fields, and `nc` is the number of components in the model.

This is the macro function that will call `evaluate_field(system,field,profiles)` for each field in the system.
"""
function evaluate_field(system::DFTSystem, ρ)
    fields = system.fields
    nf = length_fields(system)
    # nvf = length(filter(x -> typeof(x) <: VectorField, fields))
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    n = zeros(Float64,ngrid...,nf,nb)

    i = 1
    idxf = 1
    while i <= nf
        if typeof(fields[idxf]) <: ScalarField
            selectdim(n,nd+1,i) .= evaluate_field(system,fields[idxf], ρ)
            i += 1
        elseif typeof(fields[idxf]) <: VectorField
            selectdim(n,nd+1,i:i+nd-1) .= evaluate_field(system,fields[idxf], ρ)
            # selectdim(n,nd+1,i) .= dropdims(sum(selectdim(nV,nd+1,iv).^2,dims=nd+2),dims=nd+2)
            i += nd
        end
        idxf += 1
    end
    
    return n
end

"""
    integrate_field(system::DFTSystem, δf, ρ)

This function will obtain, for all fields, the functional derivative for each species / bead. The output will be a 2D array with the dimensions `(ngrid,nb)`, where `ngrid` is the number of grid points, and `nb` is the number of beads in the model.

This is the macro function that will call `integrate_field(system,field,δf,ρ)` for each field in the system.
"""
function integrate_field(system::DFTSystem, δf)
    fields = system.fields

    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb  = size(δf,nd+2)
    nf = length_fields(system)

    δFδρ_res = zeros(ngrid...,nb)
    iv = 1

    j = 1
    idxf = 1
    while j <= nf      
        if typeof(fields[idxf]) <: ScalarField
            δFδρ_res += integrate_field(system,fields[idxf],selectdim(δf,nd+1,j))
            j += 1
        elseif typeof(fields[idxf]) <: VectorField
            δFδρ_res += integrate_field(system,fields[idxf],selectdim(δf,nd+1,j:j+nd-1))
            j += nd
        end
        idxf += 1
    end

    return δFδρ_res
end