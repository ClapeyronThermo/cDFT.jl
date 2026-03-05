import Clapeyron: DHModel, screening_length, dh_term, dielectric_constant, get_sigma


function screening_length(model::DHModel, V, T, z, Z, _ϵ_r = dielectric_constant(model.RSPmodel, V, T, z))
    s = e_c*e_c/(ϵ_0*_ϵ_r*k_B*T)
    I = N_A*@sum(z[i]*Z[i]^2)
    return sqrt((s*I))
end

"""
    DH(components::Vector{String})

The PC-SAFT equation of state developed by Gross and Sadowski (2001). Our DFT implementation follows the work of Sauer and Gross (2017) which uses a Weighted Density Functional approach and does not use a chain propagator. The only additional information required in `DHSpecies` is the bead size at a given temperature.

The bulk model can be obtained from Clapeyron. 
"""
DH

struct DHSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    charges::Vector{Float64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
end

"""
    get_fields(model::EoSModel, species::DFTSpecies, structure::DFTStructure)

For a given `model`, obtain all of the fields that will be needed to perform the DFT calculation. This function should return a vector of `DFTField`s.
"""
function get_fields(ionmodel::DHModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    Z = species.charges
    ω = structure_ω(structure)
    κ = screening_length(ionmodel, 1., T, ρbulk, Z, dielectric_constant(ionmodel.RSPmodel, 1., T, ρbulk))
    d = species.size
    return [SWeightedDensity(:∫ρdz,d/2 .+ 1/κ,ω,ngrid,device)]
end


"""
    get_species(model::EoSModel, structure::DFTStructure)

For a given `model` and `structure`, define the relevant parameters for each species. These structs will contain additional information not present by default in the inital `model`, such as the bead size, the number of beads and the connectivity of the beads.
"""
function get_species(ionmodel::DHModel,model::EoSModel,charges::Vector{Int64},structure::DFTStructure)
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk
    size = get_sigma(ionmodel, 1., T, ρbulk, model)
    nc = length(ionmodel)
    return DHSpecies(ones(Int64,nc),charges,size,ρbulk)
end

"""
    get_propagator(model::EoSModel, species::DFTSpecies, structure::DFTStructure)

For a given `model`, define the relevant propagator. 
"""
function get_propagator(model::DHModel, species::DFTSpecies, structure::DFTStructure)
    return IdealPropagator()
end


function f_res(system::ElectrolyteDFTSystem, model::DHModel,n)
    return f_dh(system,model,n)
end

function f_dh(system::ElectrolyteDFTSystem, model::DHModel,n)
    (_,T) = system.structure.conditions
    σ = system.ion_species.size
    Z = system.ion_species.charges
    ρ = n ./ system.fields[end].width /2 ./ N_A

    ϵ_r = dielectric_constant(model.RSPmodel, 1., T, ρ)
    
    κ = screening_length(model, 1., T, ρ, Z, ϵ_r)
    res = zero(Base.promote_eltype(κ,σ))
    count = 0
    nc = length(model)
    for i in 1:nc
        Zi = Z[i]
        if Z[i] != 0 && !iszero(Clapeyron.primalval(ρ[i]))
            count +=1
            χi = dh_term(σ[i]*κ)
            res +=ρ[i]*Zi*Zi*χi
        end
    end
    s = e_c*e_c/(4π*ϵ_0*ϵ_r*k_B*T) 
    if iszero(count)
        return -1*s*res
    end
    return -1*s*res*κ*N_A
end


"""
    length_scale(model::EoSModel)

Obtains the maximum length scale in the model and helps define the dimensions of the DFT system. This is typically equal to the size of the largest bead.
"""
function length_scale(model::DHModel)
    return maximum(model.params.sigma.values)
end

export length_scale