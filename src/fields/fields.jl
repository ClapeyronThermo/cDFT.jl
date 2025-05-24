abstract type ScalarField <: DFTField end
abstract type VectorField <: DFTField end

include("weighted_densities.jl")

_field_len(field::ScalarField,nd::Int) = 1
_field_len(field::VectorField,nd::Int) = nd

function field_indices(fields::F,nd) where F
    ix = map(Base.Fix2(_field_len,nd),fields)
    cx = cumsum(ix)
    istart = cx .-  ix .+ 1
    iend = cx
    return range.(istart,iend)
end

field_indices(system::Union{DFTSystem,DGTSystem}) = field_indices(system.fields,dimension(system))

"""
    evaluate_field(system::DFTSystem, profiles)

This function will obtain every field used in the system (listed in `system.fields`). The output will be a 3D array with the dimensions `(ngrid,nf,nc)`, where `ngrid` is the number of grid points, `nf` is the number of fields, and `nc` is the number of components in the model.

This is the macro function that will call `evaluate_field(system,field,profiles)` for each field in the system.
"""
function evaluate_field(system::Union{DFTSystem,DGTSystem}, ρ)
    fields = system.fields
    nf = length_fields(system)
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ,nd+1)
    n = zeros(Float64,ngrid...,nf,nb)

    foreach(field_indices(system),fields) do k,field        
        if field isa ScalarField
            selectdim(n,nd+1,k[1]) .= evaluate_field(system,field,ρ)
        else
            selectdim(n,nd+1,k) .= evaluate_field(system,field,ρ)
        end
    end  
    
    return n
end

"""
    integrate_field(system::DFTSystem, δf, ρ)

This function will obtain, for all fields, the functional derivative for each species / bead. The output will be a 2D array with the dimensions `(ngrid,nb)`, where `ngrid` is the number of grid points, and `nb` is the number of beads in the model.

This is the macro function that will call `integrate_field(system,field,δf,ρ)` for each field in the system.
"""
function integrate_field(system::Union{DFTSystem,DGTSystem}, δf)
    fields = system.fields
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb  = size(δf,nd+2)
    δFδρ_res = zeros(ngrid...,nb)

    foreach(field_indices(system),fields) do k,field
        if field isa ScalarField
            δFδρ_res .+= integrate_field(system,field,selectdim(δf,nd+1,k[1]))
        else
            δFδρ_res .+= integrate_field(system,field,selectdim(δf,nd+1,k))
        end
    end
    return δFδρ_res
end