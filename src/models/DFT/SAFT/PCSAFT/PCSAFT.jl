using Clapeyron: PCSAFTModel

"""
    PCSAFT(components::Vector{String})

The PC-SAFT equation of state developed by Gross and Sadowski (2001). Our DFT implementation follows the work of Sauer and Gross (2017) which uses a Weighted Density Functional approach and does not use a chain propagator. The only additional information required in `PCSAFTSpecies` is the bead size at a given temperature.

The bulk model can be obtained from Clapeyron. 
"""
PCSAFT

struct PCSAFTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

"""
    get_fields(model::EoSModel, species::DFTSpecies, structure::DFTStructure)

For a given `model`, obtain all of the fields that will be needed to perform the DFT calculation. This function should return a vector of `DFTField`s.
"""
function get_fields(model::PCSAFTModel, species::DFTSpecies, structure::DFTStructure, device::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    nc = length(model)
    ngrid = structure.ngrid
    #f = [ngrid[i]/(structure.bounds[i,2]-structure.bounds[i,1]) for i in 1:length(ngrid)]
    ω = structure_ω(structure, device, FP)
    ψ = 1.3862
    d = species.size
    return [SWeightedDensity(:ρ,zeros(nc),ω,ngrid,device),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,device),
            SWeightedDensity(:∫ρdz,d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,device)]
end


"""
    get_species(model::EoSModel, structure::DFTStructure)

For a given `model` and `structure`, define the relevant parameters for each species. These structs will contain additional information not present by default in the inital `model`, such as the bead size, the number of beads and the connectivity of the beads.
"""
function get_species(model::PCSAFTModel,structure::DFTStructure)
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    size = d(model,1e-3,temperature,ρbulk)
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), temperature, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / temperature
    nc = length(model)
    return PCSAFTSpecies(ones(Int64,nc),size,ρbulk,μres)
end

"""
    get_propagator(model::EoSModel, species::DFTSpecies, structure::DFTStructure)

For a given `model`, define the relevant propagator. 
"""
function get_propagator(model::PCSAFTModel, species::DFTSpecies, structure::DFTStructure, backend::Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return IdealPropagator()
end

"""
    length_scale(model::EoSModel)

Obtains the maximum length scale in the model and helps define the dimensions of the DFT system. This is typically equal to the size of the largest bead.
"""
function length_scale(model::SAFTModel)
    return maximum(model.params.sigma.values)
end

export length_scale

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────
# These constants are GPU-safe (no heap allocation, no Clapeyron calls).

const PCSAFT_CORR1 = (
    (0.9105631445, -0.3084016918, -0.0906148351),
    (0.6361281449,  0.1860531159,  0.4527842806),
    (2.6861347891, -2.5030047259,  0.5962700728),
    (-26.547362491, 21.419793629, -1.7241829131),
    (97.759208784, -65.25588533,  -4.1302112531),
    (-159.59154087, 83.318680481,  13.77663187),
    (91.297774084, -33.74692293,  -8.6728470368)
)
const PCSAFT_CORR2 = (
    (0.7240946941, -0.5755498075,  0.0976883116),
    (2.2382791861,  0.6995095521, -0.2557574982),
    (-4.0025849485, 3.892567339,  -9.155856153),
    (-21.003576815,-17.215471648,  20.642075974),
    (26.855641363, 192.67226447,  -38.804430052),
    (206.55133841,-161.82646165,   93.626774077),
    (-355.60235612,-165.20769346, -29.666905585)
)

"""
Pointwise residual free energy for PC-SAFT, written in Enzyme/KernelAbstractions-compatible style.
`out[kk]` accumulates the scalar integrand at grid point `kk`.
All model parameters are unpacked from `params` (a NamedTuple of device-adapted arrays).
"""
@inline function f_res(::Type{M}, kk, out, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PCSAFTModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc  = f_hc(M, kk, n, params, T, Val(NC), Val(ND))
    res_disp, _, _ = f_disp(M, kk, n, params, T, Val(NC), Val(ND))
    res_assoc = _assoc_or_zero(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_hc + res_disp + res_assoc
    return nothing
end

"""
PC-SAFT hard-chain contribution at grid point `kk`.
Field layout assumed: field 1 = ρ (unweighted), field 4+ND = ρ̄hc, field 5+ND = λ.
"""
@inline function f_hc(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PCSAFTModel}
    FP  = eltype(n)
    m   = params.m
    HSd = params.HSd
    idx_ζ = 4 + ND;  idx_λ = 5 + ND
    ζ₃=zero(FP); ζ₂=zero(FP)
    @inbounds for i in 1:NC
        mi=m[i]; di=HSd[i]; ρ̄hci=n[kk, idx_ζ, i]
        ζ₃ += mi * ρ̄hci;  ζ₂ += mi * ρ̄hci / di
    end
    ζ₃ /= 8;  ζ₂ /= 8
    inv1ζ₃ = one(FP)/(one(FP)-ζ₃)
    res_hc = zero(FP)
    @inbounds for i in 1:NC
        ρi  = n[kk, 1, i]
        λ   = n[kk, idx_λ, i] / (2*HSd[i])
        ydd = inv1ζ₃ + FP(1.5)*HSd[i]*ζ₂*inv1ζ₃*inv1ζ₃ +
              FP(0.5)*HSd[i]*HSd[i]*ζ₂*ζ₂*inv1ζ₃*inv1ζ₃*inv1ζ₃
        res_hc += -ρi * (m[i]-1) * Base.log(abs(ydd*λ/ρi))
    end
    return res_hc
end

"""
PC-SAFT dispersion contribution at grid point `kk`. Field 6+ND is the dispersion density.
Returns `(res_disp, m̄, ηd)` — m̄ and ηd are reused by PCP-SAFT for the polar term.
"""
@inline function f_disp(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: PCSAFTModel}
    FP     = eltype(n)
    m      = params.m
    HSd    = params.HSd
    sigma  = params.sigma
    epsilon = params.epsilon
    ψ      = FP(1.3862)
    idx_ρz = 6 + ND
    factor = 3 / (4*ψ*ψ*ψ*π)

    ρ̄z_sum=zero(FP); m̄_top=zero(FP); η_sum=zero(FP)
    @inbounds for i in 1:NC
        ρ̄zi = n[kk, idx_ρz, i] * factor / (HSd[i]*HSd[i]*HSd[i])
        ρ̄z_sum += ρ̄zi
        m̄_top  += ρ̄zi * m[i]
        η_sum  += m[i] * ρ̄zi * HSd[i]*HSd[i]*HSd[i]
    end
    m̄  = m̄_top / ρ̄z_sum
    ηd = η_sum * π / 6

    m2ϵσ3_1=zero(FP); m2ϵσ3_2=zero(FP)
    @inbounds for i in 1:NC
        ρzi = n[kk, idx_ρz, i] * factor / (HSd[i]*HSd[i]*HSd[i])
        @inbounds for j in 1:NC
            ρzj  = n[kk, idx_ρz, j] * factor / (HSd[j]*HSd[j]*HSd[j])
            cij  = ρzi * ρzj * m[i] * m[j] * sigma[i,j]*sigma[i,j]*sigma[i,j]
            eT   = epsilon[i,j] / T
            m2ϵσ3_1 += cij * eT
            m2ϵσ3_2 += cij * eT * eT
        end
    end
    ηd2    = ηd*ηd
    ηd4    = (one(FP)-ηd)^4
    inv1ηd = one(FP)/(one(FP)-ηd)
    inv2ηd = one(FP)/(2-ηd)
    C₁     = one(FP) + m̄*(8*ηd-2*ηd2)/ηd4 +
              (one(FP)-m̄)*(20*ηd-27*ηd2+12*(ηd*ηd2)-2*(ηd2*ηd2)) *
              inv1ηd*inv1ηd*inv2ηd*inv2ηd
    I₁     = I_lite(PCSAFT_CORR1, m̄, ηd)
    I₂     = I_lite(PCSAFT_CORR2, m̄, ηd)
    res_disp = -2*π*I₁*m2ϵσ3_1 - π*m̄*I₂*m2ϵσ3_2 / C₁
    return res_disp, m̄, ηd
end

@inline function I_lite(corr, m̄, η)
    T  = typeof(m̄)
    res = zero(T)
    m2 = (m̄ - 1) / m̄
    m3 = m2 * (m̄ - 2) / m̄
    c1 = corr[1]; res += (T(c1[1]) + m2*T(c1[2]) + m3*T(c1[3]))
    c2 = corr[2]; res += (T(c2[1]) + m2*T(c2[2]) + m3*T(c2[3])) * η
    c3 = corr[3]; res += (T(c3[1]) + m2*T(c3[2]) + m3*T(c3[3])) * η * η
    c4 = corr[4]; res += (T(c4[1]) + m2*T(c4[2]) + m3*T(c4[3])) * η * η * η
    c5 = corr[5]; res += (T(c5[1]) + m2*T(c5[2]) + m3*T(c5[3])) * η * η * η * η
    c6 = corr[6]; res += (T(c6[1]) + m2*T(c6[2]) + m3*T(c6[3])) * η * η * η * η * η
    c7 = corr[7]; res += (T(c7[1]) + m2*T(c7[2]) + m3*T(c7[3])) * η * η * η * η * η * η
    return res
end

function preallocate_params(system::DFTSystem{<:PCSAFTModel})
    backend = system.options.device
    FP      = fptype(system.options)
    model   = system.model

    base = (;
        HSd     = adapt_to_device(backend, FP, system.species.size),
        m       = adapt_to_device(backend, FP, model.params.segment.values),
        sigma   = adapt_to_device(backend, FP, model.params.sigma.values),
        epsilon = adapt_to_device(backend, FP, model.params.epsilon.values),
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        (assoc_icomp_v, assoc_jcomp_v, assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params(model, system.species.size)

        # Integer index arrays stored as NTuples so Enzyme treats them as
        # truly-immutable Const data (avoids EnzymeRuntimeActivityError from
        # Const Vector aliasing). Padded to fixed max sizes (20 pairs, 20 sites,
        # 11 cumsum entries for ≤10 components).
        nc_model         = length(model)
        ia_global_v      = [n_sites_cumsum_v[assoc_icomp_v[p]] + assoc_isite_v[p] for p in 1:nn]
        jb_global_v      = [n_sites_cumsum_v[assoc_jcomp_v[p]] + assoc_jsite_v[p] for p in 1:nn]
        n_ia_v           = [n_sites_flat_v[ia_global_v[p]] for p in 1:nn]
        n_jb_v           = [n_sites_flat_v[jb_global_v[p]] for p in 1:nn]
        assoc_icomp_t    = ntuple(p -> assoc_icomp_v[p],    Val(nn))
        assoc_jcomp_t    = ntuple(p -> assoc_jcomp_v[p],    Val(nn))
        assoc_isite_t    = ntuple(p -> assoc_isite_v[p],    Val(nn))
        assoc_jsite_t    = ntuple(p -> assoc_jsite_v[p],    Val(nn))
        assoc_ia_global_t = ntuple(p -> ia_global_v[p],     Val(nn))
        assoc_jb_global_t = ntuple(p -> jb_global_v[p],     Val(nn))
        assoc_n_ia_t      = ntuple(p -> n_ia_v[p],          Val(nn))
        assoc_n_jb_t      = ntuple(p -> n_jb_v[p],          Val(nn))
        n_sites_flat_t   = ntuple(j -> n_sites_flat_v[j],   Val(total_sites))
        n_sites_cumsum_t = ntuple(i -> n_sites_cumsum_v[i], Val(nc_model + 1))

        assoc = (;
            has_assoc       = true,
            assoc_n_pairs   = Val(nn),
            assoc_n_sites   = Val(total_sites),
            assoc_icomp     = assoc_icomp_t,
            assoc_jcomp     = assoc_jcomp_t,
            assoc_isite     = assoc_isite_t,
            assoc_jsite     = assoc_jsite_t,
            assoc_ia_global = assoc_ia_global_t,
            assoc_jb_global = assoc_jb_global_t,
            assoc_n_ia      = assoc_n_ia_t,
            assoc_n_jb      = assoc_n_jb_t,
            assoc_eps       = adapt_to_device(backend, FP, assoc_eps_v),
            assoc_kap       = adapt_to_device(backend, FP, assoc_kap_v),
            assoc_sig3      = adapt_to_device(backend, FP, assoc_sig3_v),
            assoc_dij       = adapt_to_device(backend, FP, assoc_dij_v),
            n_sites_flat    = n_sites_flat_t,
            n_sites_cumsum  = n_sites_cumsum_t,
            total_sites,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    nc = length(model)
    return params, nc
end
