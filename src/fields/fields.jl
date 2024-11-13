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
    nf = length(fields)
    nvf = length(filter(x -> typeof(x) <: VectorField, fields))
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)

    n = zeros(Float64,ngrid...,nf,nb)
    nV = zeros(Float64,ngrid...,nvf,nb,nd)

    iv = 1
    for i in 1:nf
        if typeof(fields[i]) <: ScalarField
            selectdim(n,nd+1,i) .= evaluate_field(system,fields[i], ρ)
        elseif typeof(fields[i]) <: VectorField
            selectdim(nV,nd+1,iv) .= evaluate_field(system,fields[i], ρ)
            selectdim(n,nd+1,i) .= sum(selectdim(nV,nd+1,iv).^2,dims=nd+2)
            iv += 1
        end
    end
    
    return n, nV
end

"""
    integrate_field(system::DFTSystem, δf, ρ)

This function will obtain, for all fields, the functional derivative for each species / bead. The output will be a 2D array with the dimensions `(ngrid,nb)`, where `ngrid` is the number of grid points, and `nb` is the number of beads in the model.

This is the macro function that will call `integrate_field(system,field,δf,ρ)` for each field in the system.
"""
function integrate_field(system::DFTSystem, δf, nV)
    fields = system.fields

    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb  = size(nV,nd+2)
    nf = length(fields)

    δFδρ_res = zeros(ngrid...,nb)
    iv = 1

    for j in 1:nf        
        if typeof(fields[j]) <: ScalarField
            δFδρ_res += integrate_field(system,fields[j],selectdim(δf,nd+1,j))
        elseif typeof(fields[j]) <: VectorField
            δFδρ_res += integrate_field(system,fields[j],selectdim(δf,nd+1,j),selectdim(nV,nd+1,iv))
            iv += 1
        end
    end

    return δFδρ_res
end