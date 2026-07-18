struct ZuoStenbyGradientParam <: EoSParam
    beta::Union{PairParam{Float64},Nothing}
end

abstract type ZuoStenbyGradientModel <: GradientModel end

"""
    ZuoStenbyGradient <: GradientModel

`GradientModel` for use with `DGTSystem`, in which the influence parameter κ follows the
Zuo & Stenby (1996) correlation for cubic equations of state (J. Colloid Interface Sci. 182,
126-132):

```
κᵢ(T) = aᵢ(T) · bᵢ^(2/3) · Aᵢ · (1 - T/Tc,ᵢ)^Bᵢ,   Aᵢ = e1 + e2·ωᵢ,   Bᵢ = e3 + e4·ωᵢ + e5·ωᵢ²
```

`aᵢ(T)`, `bᵢ`, `ωᵢ` (acentric factor) and `Tc,ᵢ` are all read directly from the bulk cubic EoS
`model` passed into `kappa` -- no separate per-species database is needed (unlike
`ConstGradient`). The universal constants `e1..e5` are substance-independent but *do* depend
on which cubic EoS family the bulk `model` is: separate fitted sets are dispatched for PR, SRK
(i.e. `RK` with `SoaveAlpha`) and Patel-Teja, taken directly from Table 1 of the paper.
Species pairs are combined via the same Lorentz-Berthelot-style combining rule as
`ConstGradient` (optionally with a binary correction `beta`).

Zuo & Stenby's `a`/`b` are on a per-molecule (number-density) basis, not Clapeyron's native
per-mole convention -- `kappa` converts via `a/N_A^2`, `b/N_A` before applying eq. 6, then
`/k_B` to match `preallocate_params`'s expected units (see inline comments in `kappa` for the
full derivation).
"""
struct ZuoStenbyGradient <: ZuoStenbyGradientModel
    species::Array{String,1}
    params::ZuoStenbyGradientParam
    references::Array{String,1}
end

export ZuoStenbyGradient

"""
    ZuoStenbyGradient(components::Array{String,1}; beta = nothing)

Construct a `ZuoStenbyGradient` gradient model for the given `components`. `beta`, if given,
is a `PairParam{Float64}` binary correction applied on top of the geometric-mean combining
rule (see `kappa` below); it defaults to no correction.
"""
function ZuoStenbyGradient(components::Array{String,1}; beta::Union{PairParam{Float64},Nothing} = nothing)
    # NOTE(pjwalker): journal/volume/pages confirmed (J. Colloid Interface Sci. 1996, 182,
    # 126-132) but the DOI itself hasn't been independently verified -- double check before
    # this reference string ships anywhere citable.
    return ZuoStenbyGradient(components, ZuoStenbyGradientParam(beta), ["Zuo, Y.-X.; Stenby, E. H. J. Colloid Interface Sci. 1996, 182, 126-132"])
end

# Table 1 (Zuo & Stenby, 1996): (e1, e2, e3, e4, e5) per cubic EoS family.
_zuo_stenby_constants(::Clapeyron.PRModel) = (0.23412, -0.04838, -0.65416, 0.84064, -0.98284)
_zuo_stenby_constants(::Clapeyron.RK{<:Any,Clapeyron.SoaveAlpha,<:Any,<:Any}) = (0.28367, -0.05164, -0.81594, 1.06810, -1.11470)
_zuo_stenby_constants(::Clapeyron.PatelTejaModel) = (0.24674, -0.08480, -0.87215, 1.57290, -0.97265)
_zuo_stenby_constants(model) = error(
    "ZuoStenbyGradient has no fitted (e1..e5) constants for $(typeof(model)) -- only PR, " *
    "SRK (RK+SoaveAlpha) and Patel-Teja are supported (Table 1, Zuo & Stenby 1996).")

function kappa(gradient::ZuoStenbyGradientModel, model, T, ρ̄)
    pures = split_model(model)
    κ_values = map(pures) do pure
        e1, e2, e3, e4, e5 = _zuo_stenby_constants(pure)
        ω = Clapeyron.acentric_factor(pure)
        Tc = crit_pure(pure)[1]
        Tr = T / Tc
        a, b, _ = Clapeyron.cubic_ab(pure, one(T), T, [1.0])
        A = e1 + e2*ω
        B = e3 + e4*ω + e5*ω^2
        # Zuo & Stenby (1996) work in NUMBER densities, not molar densities (confirmed by
        # the user) -- so eq. 6's a/b must be per-molecule, not Clapeyron's native per-mole
        # a [Pa*m^6/mol^2], b [m^3/mol]. Converting: a_molecule = a/N_A^2, b_molecule = b/N_A
        # -- both fully lose their mol-dependence (N_A has units mol^-1 in strict SI, so
        # e.g. a's mol^-2 cancels exactly against N_A^2's mol^-2), landing on plain
        # a_molecule*b_molecule^(2/3) = J*m^5 with no leftover mole exponent (unlike the
        # earlier attempt using molar a/b directly, which came out as J*m^5/mol^(8/3) --
        # a real mismatch, not just a unit-system choice).
        #
        # `/k_B` then converts this number-density-native κ [J*m^5] into the "raw" value
        # `preallocate_params` expects (K*m^5, before its own `/L^3` reduction) -- derived
        # by equating dgt.jl's actual grad_term/T computation (which itself already
        # operates in the same number-density/particle-count reduced convention, per
        # n[kk,1,i]=ρ*N_A*L^3) against the f_res = F*L^3/(k_B*T*V) contract. No extra N_A
        # factor needed here specifically because the a/N_A^2, b/N_A conversion above
        # already absorbed it.
        κ_number = a * b^(2/3) * A * (1 - Tr)^B / N_A^(8/3)
        κ_number / k_B / R̄
    end
    κT = SingleParam("kappa", gradient.species, κ_values)
    return epsilon_LorentzBerthelot(κT, gradient.params.beta).values
end
