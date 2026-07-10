import Clapeyron: DHModel, screening_length, dielectric_constant, get_sigma


function screening_length(model::DHModel, V, T, z, Z, _ϵ_r = dielectric_constant(model.RSPmodel, V, T, z))
    s = e_c*e_c/(ϵ_0*_ϵ_r*k_B*T)
    I = N_A*@sum(z[i]*Z[i]^2)
    return sqrt((s*I))
end

# Padé approximant coefficients for the Debye-Hückel χ(x) term, reproduced from
# Clapeyron's `dh_term(x)` (models/Electrolytes/Ion/DH.jl) rather than called directly:
# that function hardcodes Float64 literals internally, so it always returns Float64
# regardless of its input's type — fine for Clapeyron's own bulk (non-autodiff-critical)
# use, but inside cDFT's Enzyme-differentiated kernel this produced a type-unstable
# Union{Float32,Float64} accumulator under precision=Float32 (rejected by Enzyme's strict
# type analysis). Keeping our own FP-generic copy, with `FP`-converted coefficient tuples
# threaded through `params` (same convention as `DD_consts`/`QQ_consts`/`SAFTVRMIE_A`
# elsewhere in this codebase), keeps the whole kernel — primal AND its derivative — at a
# single consistent precision instead of silently upcasting through Float64.
const DH_TERM_consts = (
    p = (0.3333333333333333, 0.7222222222346917, 0.5314393939617089, 0.15151515152716674, 0.013227513229354138),
    q = (1.0, 2.9166666667040753, 3.181818181913183, 1.5909090909939427, 0.3535353535662185, 0.026515151518857805),
)

"""
    dh_term(x, p, q)

FP-generic reimplementation of Clapeyron's `dh_term(x)`: the Debye-Hückel χ(x) =
(log(1+x) - x + x²/2)/x³ term, evaluated via a Padé approximant near `x=0` (where the
direct formula suffers catastrophic cancellation) and the direct formula otherwise. `p`/`q`
must already be `FP`-converted (see `DH_TERM_consts` above and `preallocate_params` below).
"""
@inline function dh_term(x, p, q)
    FP = typeof(x)
    if x <= FP(1.5)
        return evalpoly(x, p) / evalpoly(x, q)
    else
        x2 = x*x
        x3 = x2*x
        return (Base.log(1 + x) - x + FP(0.5)*x2) / x3
    end
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
    get_fields(ionmodel::DHModel, species, structure, device, FP; L=length_scale(ionmodel))

Builds the ion field in reduced units, mirroring PCSAFT.jl's scheme, so it stays
numerically consistent with the neutral model's own (already reduced-units) fields when
both are combined into one `ElectrolyteDFTSystem`. `L` defaults to `length_scale(ionmodel)`
for standalone use, but `ElectrolyteDFTSystem`'s constructor (base.jl) explicitly passes
`length_scale(model)` (== `length_scale(model.neutralmodel)`) instead — the neutral and ion
contributions MUST share the same `L`, since only one global `_energy_scale`/`L^3`
correction is applied to their combined `F_res`. See PCSAFT.jl's `get_fields` docstring for
the overall reduced-units scheme, and `f_dh`'s docstring below for how the compensating
`N_A*L^3` factor is threaded through this model's own free-energy kernel (which — unlike
the SAFT-family models — reads `n[]` directly rather than via an `HSd^3`-normalized ratio,
so it needs its own explicit compensation rather than an automatic cancellation). The
resolved `L` (a plain `Float64`, not `ionmodel` itself) is passed straight through to
`SWeightedDensity` in place of a model — see `length_scale(::Real)`'s docstring
(`weighted_densities.jl`) for the dispatch trick that makes this work.
"""
function get_fields(ionmodel::DHModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64; L=length_scale(ionmodel)) where FP<:AbstractFloat
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    Z = species.charges
    κ = screening_length(ionmodel, 1., temperature, ρbulk, Z, dielectric_constant(ionmodel.RSPmodel, 1., temperature, ρbulk))
    d = species.size
    ω = structure_ω(structure, device, FP)
    width = (d/2 .+ 1/κ) ./ L
    # Pass the already-resolved shared L (a plain Float64, not ionmodel) — SWeightedDensity's
    # internal `length_scale(model)` call dispatches to the `length_scale(::Real)` identity
    # method for it. See that method's docstring for why this must be the neutral model's L,
    # not `length_scale(ionmodel)`.
    return [SWeightedDensity(:∫ρdz,width,ω,ngrid,device,L)]
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

    # Shared with the neutral model's own reduced-units L (length_scale(system.model) ==
    # length_scale(system.model.neutralmodel)) — see get_fields's docstring above.
    L = length_scale(system.model)

    nc        = length(model)
    Z_vec     = system.ion_species.charges
    σ_vec     = system.ion_species.size
    width_vec = last(system.fields).width

    Z_t = ntuple(i -> i <= nc ? FP(Z_vec[i]) : zero(FP), Val(10))
    σ_t = ntuple(i -> i <= nc ? FP(σ_vec[i]) : zero(FP), Val(10))
    w_t = ntuple(i -> i <= nc ? FP(width_vec[i]) : zero(FP), Val(10))

    dh_term_p = ntuple(i -> FP(DH_TERM_consts.p[i]), 5)
    dh_term_q = ntuple(i -> FP(DH_TERM_consts.q[i]), 6)

    return (;
        dh_eps_r      = eps_r,
        dh_Z          = Z_t,
        dh_sigma      = σ_t,
        dh_width      = w_t,
        dh_L          = FP(L),
        dh_term_p     = dh_term_p,
        dh_term_q     = dh_term_q,
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

`_NA` uses `N_A*L^3` (not raw `N_A`) to invert `get_fields`'s `density_scale=L`
compensation on `n[]` — the same `NA=N_A*density_scale^3` factor `evaluate_field!` used to
build it. This is the SAFT-family models' `n[]/HSd^3`-style cancellation done explicitly:
there's no reduced length cubed in this model's own formula to cancel the inflation
automatically, so `ρi`/`I`/`κ`/`res` are recovered at their TRUE, L-independent physical
values here — required since `κ` (screening length) and `dh_term` are physically meaningful
quantities, not just numerically-convenient intermediates. The *final* multiply-back-out by
`N_A*L^3` (not raw `N_A`) then re-inflates the RETURNED value by `L^3`, matching the
neutral model's own "L^3-too-large" convention (see PCSAFT.jl's `get_fields` docstring) so
the two contributions stay on the same footing when summed in `f_res(::Type{<:ElectrolyteModel},...)`.
`dh_sigma` is deliberately left unscaled (raw ion diameter) so `σi*κ` — a genuine physical
dimensionless product `dh_term` depends on — stays correct.
"""
@inline function f_dh(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{NF_NEUTRAL}) where {M, NC, NF_NEUTRAL}
    FP    = eltype(n)
    F_dh  = NF_NEUTRAL + 1
    ε_r   = params.dh_eps_r
    L3    = params.dh_L * params.dh_L * params.dh_L
    _NA   = FP(N_A) * L3
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
    κ  = sqrt(s0 * FP(N_A) * I)

    res = zero(FP)
    @inbounds for i in 1:NC
        Zi = _nti(params.dh_Z, i)
        wi = _nti(params.dh_width, i)
        σi = _nti(params.dh_sigma, i)
        ρi = n[kk, F_dh, i] / (wi * 2) / _NA
        χi = dh_term(σi * κ, params.dh_term_p, params.dh_term_q)
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