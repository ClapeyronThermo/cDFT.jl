import Clapeyron: DHModel, screening_length, dh_term, dielectric_constant, get_sigma


function screening_length(model::DHModel, V, T, z, Z, _ϵ_r = dielectric_constant(model.RSPmodel, V, T, z))
    s = e_c*e_c/(ϵ_0*_ϵ_r*k_B*T)
    I = N_A*@sum(z[i]*Z[i]^2)
    return sqrt((s*I))
end

"""
    DH(components::Vector{String})

The (restricted primitive model) Debye-Hückel ion-ion electrostatic correction. This is
used as the `ionmodel` of a Clapeyron `ElectrolyteModel` (e.g. `ePCSAFT`), together with a
neutral bulk model, to build an [`ElectrolyteDFTSystem`](@ref cDFT.ElectrolyteDFTSystem).
No chain propagator is required — ions are treated with an `IdealPropagator`.

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
function get_fields(ionmodel::DHModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    Z = species.charges
    ω = structure_ω(structure, device, FP)
    κ = screening_length(ionmodel, 1., temperature, ρbulk, Z, dielectric_constant(ionmodel.RSPmodel, 1., temperature, ρbulk))
    d = species.size
    return [SWeightedDensity(:∫ρdz,d/2 .+ 1/κ,ω,ngrid,device)]
end


"""
    get_species(model::EoSModel, structure::DFTStructure)

For a given `model` and `structure`, define the relevant parameters for each species. These structs will contain additional information not present by default in the inital `model`, such as the bead size, the number of beads and the connectivity of the beads.
"""
function get_species(ionmodel::DHModel,model::EoSModel,charges::Vector{Int64},structure::DFTStructure)
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    size = get_sigma(ionmodel, 1., temperature, ρbulk, model)
    nc = length(ionmodel)
    return DHSpecies(ones(Int64,nc),charges,size,ρbulk)
end

"""
    get_propagator(model::EoSModel, species::DFTSpecies, structure::DFTStructure)

For a given `model`, define the relevant propagator. 
"""
function get_propagator(model::DHModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return IdealPropagator()
end

function preallocate_params(system::ElectrolyteDFTSystem, model::DHModel)
    nd         = dimension(system)
    FP         = fptype(system.options)
    NF_neutral = compute_field_len(Base.front(system.fields), nd)
    temperature = system.structure.conditions[2]
    ρbulk_ion  = system.ion_species.bulk_density
    eps_r      = FP(dielectric_constant(model.RSPmodel, 1., temperature, ρbulk_ion))

    nc        = length(model)
    Z_vec     = system.ion_species.charges
    σ_vec     = system.ion_species.size
    width_vec = last(system.fields).width

    Z_t = ntuple(i -> i <= nc ? FP(Z_vec[i]) : zero(FP), Val(10))
    σ_t = ntuple(i -> i <= nc ? FP(σ_vec[i]) : zero(FP), Val(10))
    w_t = ntuple(i -> i <= nc ? FP(width_vec[i]) : zero(FP), Val(10))

    return (;
        dh_eps_r      = eps_r,
        dh_Z          = Z_t,
        dh_sigma      = σ_t,
        dh_width      = w_t,
        dh_nf_neutral = Val(NF_neutral),
    )
end

@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: DHModel}
    out[kk] += f_dh(M, kk, n, params, T, Val(NC), params.dh_nf_neutral)
    return nothing
end

"""
GPU/Enzyme-compatible Debye-Hückel free energy density at grid point `kk`.

`NF_NEUTRAL` is the number of neutral-model field slots in `n`; the DH field
is at index `NF_NEUTRAL + 1`.  Neutral components (Z=0) contribute zero to
`I` and `res` automatically via the Zᵢ² factor — no branching needed.
`ε_r` is pre-computed at bulk density and stored in `params.dh_eps_r`.
"""
@inline function f_dh(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{NF_NEUTRAL}) where {M, NC, NF_NEUTRAL}
    FP    = eltype(n)
    F_dh  = NF_NEUTRAL + 1
    ε_r   = params.dh_eps_r
    _NA   = FP(N_A)
    _ec   = FP(e_c)
    _kB   = FP(k_B)
    _ϵ0   = FP(ϵ_0)

    I = zero(FP)
    @inbounds for i in 1:NC
        Zi = _nti(params.dh_Z, i)
        wi = _nti(params.dh_width, i)
        ρi = n[kk, F_dh, i] / (wi * 2) / _NA
        I += ρi * Zi * Zi
    end

    s0 = _ec * _ec / (_ϵ0 * ε_r * _kB * T)
    κ  = sqrt(s0 * _NA * I)

    res = zero(FP)
    @inbounds for i in 1:NC
        Zi = _nti(params.dh_Z, i)
        wi = _nti(params.dh_width, i)
        σi = _nti(params.dh_sigma, i)
        ρi = n[kk, F_dh, i] / (wi * 2) / _NA
        χi = dh_term(σi * κ)
        res += ρi * Zi * Zi * χi
    end

    s = _ec * _ec / (4 * π * _ϵ0 * ε_r * _kB * T)
    return -s * res * κ * _NA
end

"""
    length_scale(model::EoSModel)

Obtains the maximum length scale in the model and helps define the dimensions of the DFT system. This is typically equal to the size of the largest bead.
"""
function length_scale(model::DHModel)
    return maximum(model.params.sigma.values)
end

export length_scale