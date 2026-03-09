import Clapeyron: PeTS, d_pets, PeTS_A, PeTS_B

"""
    PeTS(components::Vector{String})

The PeTS equation of state developed by Langenbach (2017). This is an unpublished approach which uses a Weighted Density Functional approach and does not use a chain propagator.

The bulk model can be obtained from Clapeyron. 
"""
PeTS

struct PeTSSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::PeTSModel,structure::DFTStructure)
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk
    T̄ = T./diagvalues(model.params.epsilon.values)
    size = d_pets.(T̄).*diagvalues(model.params.sigma.values)
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T
    nc = length(model)
    return PeTSSpecies(ones(Int64,nc),size,ρbulk,μres)
end

function length_scale(model::PeTSModel)
    return maximum(model.params.sigma.values)
end

function get_fields(model::PeTSModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    nc = length(model)
    ψ = 1.21

    ω = structure_ω(structure, device)
    d = species.size
    ngrid = structure.ngrid
    
    return [SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,ψ*d,ω,ngrid,device)]
end

function get_propagator(model::PeTSModel, species::DFTSpecies, structure::DFTStructure)
    return IdealPropagator()
end

function f_res(system::DFTSystem, model::PeTSModel,n)
    return f_hs(system,model,n[1,:],n[2,:],n[3,:])+f_pert(system,model,n[4,:])
end

function f_pert(system::DFTSystem, model::PeTSModel, ρ̄)
    species = system.species
    T = system.structure.conditions[2]
    T̄ = T./diagvalues(model.params.epsilon.values)[1]
    m = model.params.segment.values
    σ = diagvalues(model.params.sigma.values)

    ψ = 1.21
    HSd = species.size

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
    ρ̃  = @sum(ρ̄[i]*m[i]*σ[i]^3)
    η = π/6*@sum(ρ̄[i]*m[i]*HSd[i]^3)
    ∑ρ̄ = sum(ρ̄)
    I1 = evalpoly(η,PeTS_A)
    I2 = evalpoly(η,PeTS_B)
    ã1 = -2*π*ρ̃ *I1/ T̄
    ã2 = -π*ρ̃*I2*(1 + 2*η*(4 - η)/(1 - η)^4)^-1 / T̄^2

    return (ã1 + ã2)*∑ρ̄
end