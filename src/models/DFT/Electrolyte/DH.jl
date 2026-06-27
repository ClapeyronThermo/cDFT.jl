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
    ω = structure_ω(structure, device)
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
function get_propagator(model::DHModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    return IdealPropagator()
end

function preallocate_params(system::ElectrolyteDFTSystem, model::DHModel)
    nd         = dimension(system)
    NF_neutral = compute_field_len(Base.front(system.fields), nd)
    T          = Float64(system.structure.conditions[2])
    ρbulk_ion  = system.ion_species.bulk_density
    eps_r      = Float64(dielectric_constant(model.RSPmodel, 1., T, ρbulk_ion))

    nc        = length(model)
    Z_vec     = system.ion_species.charges
    σ_vec     = system.ion_species.size
    width_vec = last(system.fields).width

    Z_t = ntuple(i -> i <= nc ? Float64(Z_vec[i]) : 0.0, Val(10))
    σ_t = ntuple(i -> i <= nc ? σ_vec[i]           : 0.0, Val(10))
    w_t = ntuple(i -> i <= nc ? width_vec[i]        : 0.0, Val(10))

    return (;
        dh_eps_r      = eps_r,
        dh_Z          = Z_t,
        dh_sigma      = σ_t,
        dh_width      = w_t,
        dh_nf_neutral = Val(NF_neutral),
    )
end

@inline function f_res(out, n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: DHModel}
    out[kk] += f_dh(n, params, T, kk, Val(NC), params.dh_nf_neutral)
    return nothing
end

"""
GPU/Enzyme-compatible Debye-Hückel free energy density at grid point `kk`.

`NF_NEUTRAL` is the number of neutral-model field slots in `n`; the DH field
is at index `NF_NEUTRAL + 1`.  Neutral components (Z=0) contribute zero to
`I` and `res` automatically via the Zᵢ² factor — no branching needed.
`ε_r` is pre-computed at bulk density and stored in `params.dh_eps_r`.
"""
@inline function f_dh(n, params, T, kk, ::Val{NC}, ::Val{NF_NEUTRAL}) where {NC, NF_NEUTRAL}
    _pi   = 3.141592653589793
    eps_v = 1e-30
    F_dh  = NF_NEUTRAL + 1
    ε_r   = params.dh_eps_r

    I = 0.0
    @inbounds for i in 1:NC
        Zi = _nti(params.dh_Z, i)
        wi = _nti(params.dh_width, i)
        ρi = n[kk, F_dh, i] / (wi * 2 + eps_v) / N_A
        I += ρi * Zi * Zi
    end

    s0 = e_c * e_c / (ϵ_0 * ε_r * k_B * T)
    κ  = sqrt(s0 * N_A * I + eps_v)

    res = 0.0
    @inbounds for i in 1:NC
        Zi = _nti(params.dh_Z, i)
        wi = _nti(params.dh_width, i)
        σi = _nti(params.dh_sigma, i)
        ρi = n[kk, F_dh, i] / (wi * 2 + eps_v) / N_A
        χi = dh_term(σi * κ)
        res += ρi * Zi * Zi * χi
    end

    s = e_c * e_c / (4 * _pi * ϵ_0 * ε_r * k_B * T)
    return -s * res * κ * N_A
end

"""
    length_scale(model::EoSModel)

Obtains the maximum length scale in the model and helps define the dimensions of the DFT system. This is typically equal to the size of the largest bead.
"""
function length_scale(model::DHModel)
    return maximum(model.params.sigma.values)
end

export length_scale