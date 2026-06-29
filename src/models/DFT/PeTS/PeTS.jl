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

function get_propagator(model::PeTSModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    return IdealPropagator()
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for PeTS: FMT hard-sphere + Barker–Henderson perturbation.

Field layout (for ND spatial dimensions):
  1        : ∫ρdz  with 0.5*d  → used to build n₀, n₁, n₂ FMT moments
  2        : ∫ρz²dz with 0.5*d → n₃ (volume fraction)
  3..2+ND  : ∫ρzdz with 0.5*d  → vector nᵥ
  3+ND     : ∫ρz²dz with ψ*d   → ρ̄ for perturbation term
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PeTSModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(1))
    res_pert = f_pert(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_pert
    return nothing
end

"""
PeTS Barker–Henderson perturbation contribution at grid point `kk`.
"""
@inline function f_pert(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PeTSModel}
    HSd = params.HSd
    m   = params.m
    σ   = params.sigma
    ϵ   = params.epsilon

    ψ     = 1.21
    T̄     = T / ϵ[1]
    idx_ρ̄ = 3 + ND
    ρ̃     = 0.0; η_sum = 0.0; ∑ρ̄ = 0.0
    @inbounds for i in 1:NC
        ρ̄zi  = n[kk, idx_ρ̄, i] * 3.0 / (4.0*ψ*ψ*ψ*HSd[i]*HSd[i]*HSd[i]*π)
        ρ̃    += ρ̄zi * m[i] * σ[i]*σ[i]*σ[i]
        η_sum += ρ̄zi * m[i] * HSd[i]*HSd[i]*HSd[i]
        ∑ρ̄   += ρ̄zi
    end
    η    = π / 6.0 * η_sum
    I1   = evalpoly(η, params.PeTS_A)
    I2   = evalpoly(η, params.PeTS_B)
    ã1   = -2.0*π*ρ̃*I1 / T̄
    inv_g = 1.0 / (1.0 + 2.0*η*(4.0 - η) / (1.0 - η)^4)
    ã2   = -π*ρ̃*I2*inv_g / (T̄*T̄)
    return (ã1 + ã2) * ∑ρ̄
end

function preallocate_params(system::DFTSystem{<:PeTSModel})
    backend = system.options.device
    params = (;
        HSd     = Adapt.adapt(backend, system.species.size),
        m       = Adapt.adapt(backend, system.model.params.segment.values),
        sigma   = Adapt.adapt(backend, diagvalues(system.model.params.sigma.values)),
        epsilon = Adapt.adapt(backend, diagvalues(system.model.params.epsilon.values)),
        PeTS_A  = PeTS_A,
        PeTS_B  = PeTS_B,
    )
    nc = length(system.model)
    return params, nc
end
