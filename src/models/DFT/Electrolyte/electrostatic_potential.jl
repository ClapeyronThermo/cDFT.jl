abstract type ElectrostaticPotentialModel <: ExternalFieldModel end

struct ElectrostaticPotential{M} <: ElectrostaticPotentialModel
    ϵ_r::Float64
    map::M
end

export ElectrostaticPotential

function ElectrostaticPotential(model::ElectrolyteModel, structure::DFTStructure, backend::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    (_, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ϵ_r = dielectric_constant(model.ionmodel.RSPmodel, 1., temperature, ρbulk)
    ngrid = structure.ngrid
    nd = length(ngrid)

    ω = structure_ω(structure, backend, FP)

    ω_norm = allocate(CPU(), FP, ngrid...)

    for kk in CartesianIndices(ngrid)
        ω_norm[kk] = norm(@view(ω[Tuple(kk)...,:]))
    end

    ω̄ = allocate(backend, FP, ngrid...)
    copyto!(ω̄,Adapt.adapt(typeof(ω̄), ω_norm))

    _c  = FP(N_A * e_c^2 / ϵ_0) / FP(ϵ_r)
    Ω   = @. (!iszero(ω̄)) / (FP(4)*π*π*ω̄^2 + iszero(ω̄)) * _c
    return ElectrostaticPotential(ϵ_r, Ω)
end

"""
    ElectrostaticPotential(model, structure::Union{DFTStructureSphr,DFTStructureCyl}, backend, FP)

Spherical/cylindrical (QDHT-based) counterpart of the Cartesian `ElectrostaticPotential`
constructor above. Reuses the same 3D-isotropic Coulomb Green's-function kernel formula
unchanged — a `z`-translation-invariant 3D charge distribution only excites the `k_z=0`
slice of the isotropic 3D Coulomb kernel, which has the same functional form, so no
rescaling is needed between the spherical and cylindrical cases. `ω̄=0` never occurs on
the QDHT grid, so the Cartesian version's zero-mode mask is unnecessary here.
"""
function ElectrostaticPotential(model::ElectrolyteModel, structure::Union{DFTStructureSphr,DFTStructureCyl}, backend::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    backend isa CPU || error("Spherical/cylindrical coordinate systems are CPU-only for now")
    (_, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ϵ_r = dielectric_constant(model.ionmodel.RSPmodel, 1., temperature, ρbulk)

    ω̄ = structure_ω(structure, backend, FP).ω̄

    _c = FP(N_A * e_c^2 / ϵ_0) / FP(ϵ_r)
    Ω  = @. _c / (FP(4)*π*π*ω̄^2)
    return ElectrostaticPotential(ϵ_r, Ω)
end


function evaluate_external_field!(structure::DFTStructure,external_field::ElectrostaticPotentialModel,model::ElectrolyteModel,ρ,δfδρ_res,P,iP,Vext)
    temperature = structure.conditions[2]
    Z = model.charge
    ngrid = structure.ngrid
    bounds = structure.bounds
    L = bounds[2] - bounds[1]
    Vol = prod(L)
    nbeads = length(Z)
    nd = length(ngrid)
    # obtain charge profiles
    for i in 1:nbeads
        # println(i)
        if i == 1
            Vext .= selectdim(ρ,nd+1,i)*Z[i]
        else
            Vext .+= selectdim(ρ,nd+1,i)*Z[i]
        end
    end

    ϵ_r = external_field.ϵ_r
    map = external_field.map
    
    convolve!(Vext, Vext, map, P, iP, Vext)

    for i in 1:nbeads
        selectdim(δfδρ_res,nd+1,i) .+= Z[i]*Vext / k_B / temperature
    end
end

function find_ψ_const(structure::DFTStructure,external_field::ElectrostaticPotentialModel,model::ElectrolyteModel,ρ::Array{Float64})
    Z = model.charge
    nbeads = length(Z)
    nd = length(structure.ngrid)
    # obtain charge profiles
    ψ0 = 0.
    while true
        q = 0.
        dq = 0.
        for i in 1:nbeads
            q += sum(selectdim(ρ,nd+1,i)*Z[i])*exp(-Z[i]*ψ0)
            dq -= sum(selectdim(ρ,nd+1,i))*Z[i]^2*exp(-Z[i]*ψ0)
        end
        ψ0 -= q/dq
        # println("ψ0 = ", ψ0, " q = ", q)
        if abs(q) < 1e-6
            break
        end
    end
    return ψ0*k_B*structure.conditions[2]
end