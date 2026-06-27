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
function get_fields(model::PCSAFTModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    nc = length(model)
    ngrid = structure.ngrid
    #f = [ngrid[i]/(structure.bounds[i,2]-structure.bounds[i,1]) for i in 1:length(ngrid)]
    ω = structure_ω(structure, device)
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
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk
    size = d(model,1e-3,T,ρbulk)
    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk/sum(ρbulk)) / Clapeyron.R̄ / T
    nc = length(model)
    return PCSAFTSpecies(ones(Int64,nc),size,ρbulk,μres)
end

"""
    get_propagator(model::EoSModel, species::DFTSpecies, structure::DFTStructure)

For a given `model`, define the relevant propagator. 
"""
function get_propagator(model::PCSAFTModel, species::DFTSpecies, structure::DFTStructure, backend::Backend)
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
# These constants and helpers are GPU-safe (no heap allocation, no Clapeyron calls).

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

@inline function I_lite(corr, m̄, η)
    res = 0.0
    m2 = (m̄ - 1.0) / m̄
    m3 = m2 * (m̄ - 2.0) / m̄
    c1 = corr[1]; res += (c1[1] + m2*c1[2] + m3*c1[3])
    c2 = corr[2]; res += (c2[1] + m2*c2[2] + m3*c2[3]) * η
    c3 = corr[3]; res += (c3[1] + m2*c3[2] + m3*c3[3]) * η * η
    c4 = corr[4]; res += (c4[1] + m2*c4[2] + m3*c4[3]) * η * η * η
    c5 = corr[5]; res += (c5[1] + m2*c5[2] + m3*c5[3]) * η * η * η * η
    c6 = corr[6]; res += (c6[1] + m2*c6[2] + m3*c6[3]) * η * η * η * η * η
    c7 = corr[7]; res += (c7[1] + m2*c7[2] + m3*c7[3]) * η * η * η * η * η * η
    return res
end

"""
PC-SAFT hard-chain contribution at grid point `kk`.
Field layout assumed: field 1 = ρ (unweighted), field 4+ND = ρ̄hc, field 5+ND = λ.
"""
@inline function f_hc(n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: PCSAFTModel}
    eps_v = 1e-15
    m   = params.m
    HSd = params.HSd
    idx_ζ = 4 + ND;  idx_λ = 5 + ND
    ζ₃=0.0; ζ₂=0.0
    @inbounds for i in 1:NC
        mi=m[i]; di=HSd[i]; ρ̄hci=n[kk, idx_ζ, i]
        ζ₃ += mi * ρ̄hci;  ζ₂ += mi * ρ̄hci / di
    end
    ζ₃ *= 0.125;  ζ₂ *= 0.125
    inv1ζ₃ = 1.0/(1.0-ζ₃+eps_v)
    res_hc = 0.0
    @inbounds for i in 1:NC
        ρi  = n[kk, 1, i]
        λ   = n[kk, idx_λ, i] / (2.0*HSd[i])
        ydd = inv1ζ₃ + 1.5*HSd[i]*ζ₂*inv1ζ₃*inv1ζ₃ +
              0.5*HSd[i]*HSd[i]*ζ₂*ζ₂*inv1ζ₃*inv1ζ₃*inv1ζ₃
        res_hc += -ρi * (m[i]-1.0) * Base.log(abs(ydd*λ/(ρi+eps_v))+eps_v)
    end
    return res_hc
end

"""
PC-SAFT dispersion contribution at grid point `kk`. Field 6+ND is the dispersion density.
Returns `(res_disp, m̄, ηd)` — m̄ and ηd are reused by PCP-SAFT for the polar term.
"""
@inline function f_disp(n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: PCSAFTModel}
    _pi    = 3.141592653589793
    eps_v  = 1e-15
    m      = params.m
    HSd    = params.HSd
    sigma  = params.sigma
    epsilon = params.epsilon
    ψ      = 1.3862
    idx_ρz = 6 + ND
    factor = 3.0 / (4.0*ψ*ψ*ψ*_pi)

    ρ̄z_sum=eps_v; m̄_top=0.0; η_sum=0.0
    @inbounds for i in 1:NC
        ρ̄zi = n[kk, idx_ρz, i] * factor / (HSd[i]*HSd[i]*HSd[i])
        ρ̄z_sum += ρ̄zi
        m̄_top  += ρ̄zi * m[i]
        η_sum  += m[i] * ρ̄zi * HSd[i]*HSd[i]*HSd[i]
    end
    m̄  = m̄_top / ρ̄z_sum
    ηd = _pi/6.0 * η_sum

    m2ϵσ3_1=0.0; m2ϵσ3_2=0.0
    @inbounds for i in 1:NC
        ρzi = n[kk, idx_ρz, i] * factor / (HSd[i]*HSd[i]*HSd[i])
        @inbounds for j in i:NC
            ρzj  = n[kk, idx_ρz, j] * factor / (HSd[j]*HSd[j]*HSd[j])
            cij  = ρzi * ρzj * m[i] * m[j] * sigma[i,j]*sigma[i,j]*sigma[i,j]
            eT   = epsilon[i,j] / (T + eps_v)
            t1   = cij * eT;  t2 = cij * eT * eT
            if i == j
                m2ϵσ3_1 += t1;       m2ϵσ3_2 += t2
            else
                m2ϵσ3_1 += 2.0*t1;   m2ϵσ3_2 += 2.0*t2
            end
        end
    end
    ηd2    = ηd*ηd
    ηd4    = (1.0-ηd+eps_v)^4
    inv1ηd = 1.0/(1.0-ηd+eps_v)
    inv2ηd = 1.0/(2.0-ηd+eps_v)
    C₁     = 1.0 + m̄*(8.0*ηd-2.0*ηd2)/ηd4 +
              (1.0-m̄)*(20.0*ηd-27.0*ηd2+12.0*(ηd*ηd2)-2.0*(ηd2*ηd2)) *
              inv1ηd*inv1ηd*inv2ηd*inv2ηd
    I₁     = I_lite(PCSAFT_CORR1, m̄, ηd)
    I₂     = I_lite(PCSAFT_CORR2, m̄, ηd)
    res_disp = -2.0*_pi*I₁*m2ϵσ3_1 - _pi*m̄*I₂*m2ϵσ3_2 / (C₁+eps_v)
    return res_disp, m̄, ηd
end

"""
Pointwise residual free energy for PC-SAFT, written in Enzyme/KernelAbstractions-compatible style.
`out[kk]` accumulates the scalar integrand at grid point `kk`.
All model parameters are unpacked from `params` (a NamedTuple of device-adapted arrays).
"""
@inline function f_res(out, n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: PCSAFTModel}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc  = f_hc(n, params, T, kk, Val(NC), Val(ND), M)
    res_disp, _, _ = f_disp(n, params, T, kk, Val(NC), Val(ND), M)
    res_assoc = _assoc_or_zero(n, params, T, kk, Val(NC), Val(ND), M)
    out[kk] = res_hs + res_hc + res_disp + res_assoc
    return nothing
end

function preallocate_params(system::DFTSystem{<:PCSAFTModel})
    backend = system.options.device
    model   = system.model

    base = (;
        HSd     = Adapt.adapt(backend, system.species.size),
        m       = Adapt.adapt(backend, model.params.segment.values),
        sigma   = Adapt.adapt(backend, model.params.sigma.values),
        epsilon = Adapt.adapt(backend, model.params.epsilon.values),
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
        nc_model = length(model)
        assoc_icomp_t    = ntuple(p -> p <= nn           ? assoc_icomp_v[p]    : 0, Val(20))
        assoc_jcomp_t    = ntuple(p -> p <= nn           ? assoc_jcomp_v[p]    : 0, Val(20))
        assoc_isite_t    = ntuple(p -> p <= nn           ? assoc_isite_v[p]    : 0, Val(20))
        assoc_jsite_t    = ntuple(p -> p <= nn           ? assoc_jsite_v[p]    : 0, Val(20))
        n_sites_flat_t   = ntuple(j -> j <= total_sites  ? n_sites_flat_v[j]   : 0, Val(20))
        n_sites_cumsum_t = ntuple(i -> i <= nc_model + 1 ? n_sites_cumsum_v[i] : 0, Val(11))

        assoc = (;
            has_assoc      = true,
            assoc_n_pairs  = Val(nn),
            assoc_icomp    = assoc_icomp_t,
            assoc_jcomp    = assoc_jcomp_t,
            assoc_isite    = assoc_isite_t,
            assoc_jsite    = assoc_jsite_t,
            assoc_eps      = Adapt.adapt(backend, assoc_eps_v),
            assoc_kap      = Adapt.adapt(backend, assoc_kap_v),
            assoc_sig3     = Adapt.adapt(backend, assoc_sig3_v),
            assoc_dij      = Adapt.adapt(backend, assoc_dij_v),
            n_sites_flat   = n_sites_flat_t,
            n_sites_cumsum = n_sites_cumsum_t,
            total_sites,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    nc = length(model)
    return params, nc
end
