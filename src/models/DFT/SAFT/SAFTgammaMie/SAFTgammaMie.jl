using Clapeyron: SAFTgammaMieModel
using Clapeyron: d_gc_av

function DFTSystem(model::SAFTgammaMieModel,structure::DFTStructure,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure, options.device)
    propagator = get_propagator(model, species, structure, options.device)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, fields, nothing, propagator, options, chunksize)
end

function DFTSystem(model::SAFTgammaMieModel,structure::DFTStructure, external_field,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure, options.device)
    propagator = get_propagator(model, species, structure, options.device)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, fields, external_field, propagator, options, chunksize)
end


struct SAFTgammaMieSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::SAFTgammaMie,structure::DFTStructure)
    (p,T) = structure.conditions
    ρbulk = structure.ρbulk 
    HSd = d(model,1e-3,T,ones(length(model.groups.flattenedgroups)))

    μres = Clapeyron.VT_chemical_potential_res(model, 1/sum(ρbulk), T, ρbulk./sum(ρbulk)) / Clapeyron.R̄ / T
    nbeads = length.(model.groups.groups)

    levels = zeros(Int, sum(nbeads))

    for i in @comps
        i_groups = model.groups.i_groups[i]
        bond_mat = Bool.(model.groups.n_intergroups[i])
        nbonds = sum(bond_mat,dims=2)[:]
        is_leaf = nbonds .== 1
        i_root = i_groups[findfirst(nbonds[i_groups] .== maximum(nbonds[i_groups]))]
        levels[i_root] = 1
    
        idx_current_level = i_root
        is_bonded = bond_mat[idx_current_level,:]
        k = 1
        while any(levels[i_groups] .== 0)
            levels[is_bonded] .= k+1
            idx_next_level = findall(levels .== k+1 .&& .!(is_leaf))
            is_bonded = (sum(bond_mat[idx_next_level,:],dims=1)[:].==1 .&& levels.==0)
            k+=1
        end
    end
    return SAFTgammaMieSpecies(nbeads,HSd,levels,ρbulk,μres)
end

function get_fields(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    nb = sum(species.nbeads)
    ngrid = structure.ngrid
    nd = dimension(structure)
    ω = structure_ω(structure, device)
    d = species.size
    λ_r = diagvalues(model.params.lambda_r.values)
    λ_a = diagvalues(model.params.lambda_a.values)
    σ   = diagvalues(model.params.sigma.values)
    C = @. λ_r / (λ_r - λ_a) * (λ_r / λ_a)^(λ_a / (λ_r - λ_a))
    x = d ./ σ
    ψ = @. cbrt(3*C*(1/(λ_a-3)-1/(λ_r-3)))
    return [SWeightedDensity(:ρ,zeros(nb),ω,ngrid,device),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,device),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,device),
            SWeightedDensity(:∫ρdz,d,ω,ngrid,device),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,device)]
end

function get_propagator(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure, device::Backend)
    return TangentHSPropagator(model, species, structure, device)
end



function expand_model(model::SAFTgammaMieModel) 
    
    nspecies = length(model)

    #Expand groups
    grouparam,ngroups_k = expand_groups(model)
    
    #Expand the sites 
    siteparams = expand_sites(model, grouparam, ngroups_k)
    params_old,vrparams_old = model.params,model.vrmodel.params
    PARAM = typeof(params_old)
    
    oldparams = PARAM(params_old.segment,params_old.shapefactor,params_old.lambda_a,params_old.lambda_r,params_old.sigma,params_old.epsilon,vrparams_old.epsilon_assoc,vrparams_old.bondvol,params_old.mixed_segment)
    #Expand the parameters
    eosparams = expand_params(oldparams, grouparam, siteparams, ngroups_k)

    #compute mixed segment
    Clapeyron.mix_segment!(eosparams.mixed_segment,grouparam,eosparams.shapefactor.values,eosparams.segment.values).values
    vreosparam_type = typeof(model.vrmodel.params)
    vreosparams = vreosparam_type(model.vrmodel.params.Mw,
                                  model.vrmodel.params.segment,
                                  model.vrmodel.params.sigma,
                                  model.vrmodel.params.lambda_a,
                                  model.vrmodel.params.lambda_r,
                                  model.vrmodel.params.epsilon,
                                  eosparams.epsilon_assoc,
                                  eosparams.bondvol)

    vrmodel = SAFTVRMie(model.vrmodel.components,
                        siteparams,
                        vreosparams,
                        model.vrmodel.idealmodel,
                        model.vrmodel.assoc_options,
                        model.vrmodel.references
                        )

    return new_model = SAFTgammaMie(model.components,
                                grouparam,
                                siteparams,
                                eosparams,
                                model.idealmodel,
                                vrmodel,
                                model.epsilon_mixing,
                                model.assoc_options,
                                model.references)
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

# Model-specific Wertheim Δ for SAFTgammaMie: VR Mie polynomial I(Tr, ρr).
# Uses params.meff (= m.*S) for ρS and params.assoc_ispec/jspec + params.epsilon_species
# (nc_spec × nc_spec VR Mie ε at species level) for the reduced temperature Tr.
@inline function _assoc_delta(p, n_pairs, n, params, T, kk, n3_mix, n2_mix, xi_mix, eps_v,
                               ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTgammaMieModel}
    p > n_pairs && return 0.0
    _pi = 3.141592653589793
    ρS = eps_v
    @inbounds for k in 1:NC
        ρS += n[kk, 3, k] * 6.0 / (_pi * params.HSd[k]^3) * params.meff[k]
    end
    σ3_x = 0.0
    @inbounds for k in 1:NC
        ρ̄k  = n[kk, 3, k] * 6.0 / (_pi * params.HSd[k]^3)
        xSk = ρ̄k * params.meff[k] / ρS
        σ3_x += xSk * xSk * params.sigma[k,k]^3
        @inbounds for l in 1:(k-1)
            ρ̄l  = n[kk, 3, l] * 6.0 / (_pi * params.HSd[l]^3)
            xSl = ρ̄l * params.meff[l] / ρS
            σ3_x += 2.0 * xSk * xSl * params.sigma[k,l]^3
        end
    end
    ρr = ρS * σ3_x
    is = _nti(params.assoc_ispec, p)
    js = _nti(params.assoc_jspec, p)
    Tr = T / (params.epsilon_species[is, js] + eps_v)
    I_val = 0.0; ρrn = 1.0
    for ni in 0:10
        row = _nti(params.VRMie_c, ni + 1); Trm = 1.0
        for mi in 0:10
            I_val += _nti(row, mi + 1) * Trm * ρrn; Trm *= Tr
        end
        ρrn *= ρr
    end
    return expm1(params.assoc_eps[p] / T) * params.assoc_kap[p] * I_val
end

"""
Pointwise residual free energy for SAFTγMie: FMT hard-sphere (with shapefactor) +
chain (groups aggregated to species level, gMie from vrmodel params) +
SAFT-VR Mie dispersion (with effective m*S segments).

Field layout (same as SAFTVRMieModel):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc  (for TangentHSPropagator chain)
  5+ND     : ∫ρdz  with d     → λ    (for TangentHSPropagator chain)
  6+ND     : ∫ρz²dz with d*ψ → ρ̄z   (dispersion)

NC here is the total number of groups (sum of nbeads per component).
"""
@inline function f_chain(n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTgammaMieModel}
    _pi   = 3.141592653589793
    eps_v = 1e-15

    HSd    = params.HSd
    meff   = params.meff
    σ      = params.sigma
    A      = params.A
    ϕ      = params.phi

    idx_ζ_c = 4 + ND
    nbeads_c = params.nbeads_comp
    HSd_s    = params.HSd_species
    m_s      = params.m_species
    σ_s      = params.sigma_species
    ϵ_s      = params.epsilon_species
    λr_s     = params.lambda_r_species_t
    λa_s     = params.lambda_a_species_t
    nc_s     = length(nbeads_c)

    ρS_c  = eps_v
    sg_idx = params.species_group_idx
    @inbounds for s in 1:nc_s
        nb_s = _nti(nbeads_c, s)
        ρ̄hc_s = 0.0
        @inbounds for j in 1:nb_s
            kg = sg_idx[s][j]
            dg = HSd[kg]
            ρ̄hc_s += n[kk, idx_ζ_c, kg] * 3.0/(4.0*_pi*dg^3)
        end
        ρ̄hc_s /= Float64(nb_s)
        ρS_c += ρ̄hc_s * m_s[s]
    end
    kρS_c = ρS_c * _pi/6.0/8.0

    ρhc_gc_total = eps_v
    @inbounds for kg in 1:NC
        ρhc_gc_total += n[kk, 1, kg]
    end
    m̄_gc = 0.0
    @inbounds for kg in 1:NC
        z_gc_kg = n[kk, 1, kg] / ρhc_gc_total
        m̄_gc += z_gc_kg * meff[kg]
    end
    m̄inv_gc = 1.0/(m̄_gc + eps_v)

    ζ_Xc = 0.0;  σ3_xc = 0.0
    @inbounds for i in 1:NC
        z_gc_i = n[kk, 1, i] / ρhc_gc_total
        x_Si_c = z_gc_i * meff[i] * m̄inv_gc
        di_c   = HSd[i]
        σ3_xc += x_Si_c*x_Si_c*σ[i,i]^3
        ζ_Xc  += kρS_c*x_Si_c*x_Si_c*(2.0*di_c)^3
        @inbounds for j in 1:(i-1)
            z_gc_j = n[kk, 1, j] / ρhc_gc_total
            x_Sj_c = z_gc_j * meff[j] * m̄inv_gc
            dj_c   = HSd[j]
            σ3_xc += 2.0*x_Si_c*x_Sj_c*σ[i,j]^3
            ζ_Xc  += 2.0*kρS_c*x_Si_c*x_Sj_c*(di_c+dj_c)^3
        end
    end
    ζstc = σ3_xc * ρS_c * _pi/6.0
    _KHSc, _∂KHSc = _KHS_fdf_kernel(ρS_c, ζ_Xc)

    res_chain = 0.0
    @inbounds for s in 1:nc_s
        nb_s = _nti(nbeads_c, s)
        ρhc_s = 0.0
        @inbounds for j in 1:nb_s
            kg = sg_idx[s][j]
            ρhc_s += n[kk, 1, kg]
        end
        ρhc_s /= Float64(nb_s)

        di_s = HSd_s[s]
        λa_c = λa_s[s][s];  λr_c = λr_s[s][s]
        _Cc  = _Cλ_kernel(λa_c, λr_c)
        x0c  = σ_s[s,s] / di_s
        ϵiic = ϵ_s[s,s]

        aS1c_a,  dS1c_a  = _aS1_fdf_kernel(λa_c,       ζ_Xc, A)
        aS1c_r,  dS1c_r  = _aS1_fdf_kernel(λr_c,       ζ_Xc, A)
        Bc_a,    dBc_a   = _B_fdf_kernel(λa_c,     x0c, ζ_Xc)
        Bc_r,    dBc_r   = _B_fdf_kernel(λr_c,     x0c, ζ_Xc)
        aS1c_2a, dS1c_2a = _aS1_fdf_kernel(2.0*λa_c,   ζ_Xc, A)
        aS1c_2r, dS1c_2r = _aS1_fdf_kernel(2.0*λr_c,   ζ_Xc, A)
        aS1c_ar, dS1c_ar = _aS1_fdf_kernel(λa_c+λr_c,  ζ_Xc, A)
        Bc_2a,   dBc_2a  = _B_fdf_kernel(2.0*λa_c, x0c, ζ_Xc)
        Bc_2r,   dBc_2r  = _B_fdf_kernel(2.0*λr_c, x0c, ζ_Xc)
        Bc_ar,   dBc_ar  = _B_fdf_kernel(λa_c+λr_c,x0c, ζ_Xc)

        ∂a1ρSc = _Cc*(x0c^λa_c*(dS1c_a+dBc_a) - x0c^λr_c*(dS1c_r+dBc_r))
        g1c    = 3.0*∂a1ρSc - _Cc*(λa_c*x0c^λa_c*(aS1c_a+Bc_a) - λr_c*x0c^λr_c*(aS1c_r+Bc_r))

        αc  = _Cc*(1.0/(λa_c-3.0) - 1.0/(λr_c-3.0))
        f1c,f2c,f3c,f4c,f5c,f6c = _f123456_kernel(αc, ϕ)
        θc  = exp(ϵiic/T) - 1.0
        γcc = 10.0*(-tanh(10.0*(0.57-αc))+1.0)*ζstc*θc*exp(-6.7*ζstc-8.0*ζstc^2)

        cb2ac = x0c^(2.0*λa_c)*(aS1c_2a+Bc_2a)
        cbarc = x0c^(λa_c+λr_c)*(aS1c_ar+Bc_ar)
        cb2rc = x0c^(2.0*λr_c)*(aS1c_2r+Bc_2r)
        ∂a2ρSc = 0.5*_Cc*_Cc*(
            ρS_c*_∂KHSc*(cb2ac - 2.0*cbarc + cb2rc)
          + _KHSc*(x0c^(2.0*λa_c)*(dS1c_2a+dBc_2a)
                 - 2.0*x0c^(λa_c+λr_c)*(dS1c_ar+dBc_ar)
                 + x0c^(2.0*λr_c)*(dS1c_2r+dBc_2r))
        )
        gMCA2c = 3.0*∂a2ρSc - _KHSc*_Cc*_Cc*(λr_c*cb2rc - (λa_c+λr_c)*cbarc + λa_c*cb2ac)
        g2c    = (1.0+γcc)*gMCA2c

        gHSc  = _gHS_kernel(x0c, ζ_Xc)
        gMiec = gHSc * exp(ϵiic/T * g1c/gHSc + (ϵiic/T)^2 * g2c/gHSc)

        ms = m_s[s]
        res_chain += ρhc_s * Base.log(abs(gMiec) + eps_v) * (ms - 1.0)
    end
    return -res_chain
end

@inline function f_res(out, n, params, T, kk,
                       ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTgammaMieModel}
    res_hs, = f_hs(n, params.meff, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_disp  = f_disp(n, params, kk, T, Val(NC), Val(ND), Val(6+ND), M)
    res_chain = f_chain(n, params, T, kk, Val(NC), Val(ND), M)
    res_assoc = _assoc_or_zero(n, params, T, kk, Val(NC), Val(ND), M)
    out[kk] = res_hs + res_chain + res_disp + res_assoc
    return nothing
end

function preallocate_params(system::DFTSystem{<:SAFTgammaMieModel})
    backend = system.options.device
    model   = system.model
    T_val  = system.structure.conditions[2]
    x_val  = system.structure.ρbulk ./ sum(system.structure.ρbulk)
    HSd_sp = d_gc_av(model, 1e-3, T_val, x_val, system.species.size)

    m_vals = model.params.segment.values
    S_vals = model.params.shapefactor.values
    meff   = m_vals .* S_vals

    nc_g = sum(system.species.nbeads)
    lr_g = model.params.lambda_r.values
    la_g = model.params.lambda_a.values
    lambda_r_t = ntuple(i -> ntuple(j -> lr_g[i,j], nc_g), nc_g)
    lambda_a_t = ntuple(i -> ntuple(j -> la_g[i,j], nc_g), nc_g)
    ncs = size(model.vrmodel.params.lambda_r.values, 1)
    lr_s = model.vrmodel.params.lambda_r.values
    la_s = model.vrmodel.params.lambda_a.values
    lambda_r_species_t = ntuple(i -> ntuple(j -> lr_s[i,j], ncs), ncs)
    lambda_a_species_t = ntuple(i -> ntuple(j -> la_s[i,j], ncs), ncs)
    nc_s_v   = length(system.species.nbeads)
    nbeads_v = system.species.nbeads
    max_nb   = maximum(nbeads_v)
    species_group_idx = ntuple(s -> ntuple(j -> (j <= nbeads_v[s] ? model.groups.i_groups[s][j] : 0), max_nb), nc_s_v)
    base = (;
        HSd                = Adapt.adapt(backend, system.species.size),
        m                  = Adapt.adapt(backend, m_vals),
        S                  = Adapt.adapt(backend, S_vals),
        meff               = Adapt.adapt(backend, meff),
        sigma              = Adapt.adapt(backend, model.params.sigma.values),
        epsilon            = Adapt.adapt(backend, model.params.epsilon.values),
        lambda_r_t         = lambda_r_t,
        lambda_a_t         = lambda_a_t,
        psi_eff            = Adapt.adapt(backend, system.fields[end].width),
        A                  = SAFTVRMIE_A,
        phi                = SAFTVRMIE_PHI,
        nbeads_comp        = ntuple(i -> system.species.nbeads[i], nc_s_v),
        HSd_species        = Adapt.adapt(backend, HSd_sp),
        m_species          = Adapt.adapt(backend, model.vrmodel.params.segment.values),
        sigma_species      = Adapt.adapt(backend, model.vrmodel.params.sigma.values),
        epsilon_species    = Adapt.adapt(backend, model.vrmodel.params.epsilon.values),
        lambda_r_species_t = lambda_r_species_t,
        lambda_a_species_t = lambda_a_species_t,
        species_group_idx  = species_group_idx,
    )

    nn = Clapeyron.assoc_pair_length(model)
    if nn > 0
        nc_model = sum(system.species.nbeads)
        (assoc_icomp_v, assoc_jcomp_v, assoc_ispec_v, assoc_jspec_v,
         assoc_isite_v, assoc_jsite_v,
         assoc_eps_v, assoc_kap_v, assoc_sig3_v, assoc_dij_v,
         n_sites_flat_v, n_sites_cumsum_v, total_sites
        ) = pack_assoc_params_gc(model, system.species.size)

        assoc_icomp_t    = ntuple(p -> p <= nn           ? assoc_icomp_v[p]    : 0, Val(20))
        assoc_jcomp_t    = ntuple(p -> p <= nn           ? assoc_jcomp_v[p]    : 0, Val(20))
        assoc_ispec_t    = ntuple(p -> p <= nn           ? assoc_ispec_v[p]    : 0, Val(20))
        assoc_jspec_t    = ntuple(p -> p <= nn           ? assoc_jspec_v[p]    : 0, Val(20))
        assoc_isite_t    = ntuple(p -> p <= nn           ? assoc_isite_v[p]    : 0, Val(20))
        assoc_jsite_t    = ntuple(p -> p <= nn           ? assoc_jsite_v[p]    : 0, Val(20))
        n_sites_flat_t   = ntuple(j -> j <= total_sites  ? n_sites_flat_v[j]   : 0, Val(20))
        n_sites_cumsum_t = ntuple(i -> i <= nc_model + 1
                                       ? n_sites_cumsum_v[i]
                                       : n_sites_cumsum_v[nc_model + 1], Val(30))

        c_mat   = SAFTVRMieconsts.c
        VRMie_c = ntuple(ni -> ntuple(mi -> c_mat[ni, mi], 11), 11)

        assoc = (;
            has_assoc      = true,
            assoc_n_pairs  = Val(nn),
            assoc_icomp    = assoc_icomp_t,
            assoc_jcomp    = assoc_jcomp_t,
            assoc_ispec    = assoc_ispec_t,
            assoc_jspec    = assoc_jspec_t,
            assoc_isite    = assoc_isite_t,
            assoc_jsite    = assoc_jsite_t,
            assoc_eps      = Adapt.adapt(backend, assoc_eps_v),
            assoc_kap      = Adapt.adapt(backend, assoc_kap_v),
            assoc_sig3     = Adapt.adapt(backend, assoc_sig3_v),
            assoc_dij      = Adapt.adapt(backend, assoc_dij_v),
            n_sites_flat   = n_sites_flat_t,
            n_sites_cumsum = n_sites_cumsum_t,
            total_sites,
            VRMie_c,
        )
        params = merge(base, assoc)
    else
        params = merge(base, (; has_assoc = false))
    end

    nc = sum(system.species.nbeads)
    return params, nc
end
