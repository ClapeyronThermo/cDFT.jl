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

function get_propagator(model::SAFTgammaMieModel, species::DFTSpecies, structure::DFTStructure, device)
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

function Δ(model::SAFTgammaMieModel, T, n, n₃, nᵥ, i, j, a, b)
    _d = d(model,1e-3,T,ones(length(model.groups.flattenedgroups)))
    _σ = model.params.sigma.values
    m = model.params.segment.values
    S = model.params.shapefactor.values
    ϵ_assoc = model.params.epsilon_assoc.values
    K = model.params.bondvol.values[i,j][a,b]
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(K))
    iszero(K) && return _0

    ρ̄ = n₃*3*2 ./(_d.^3)/π
    m = m.*S
    z = ρ̄ /sum(ρ̄)
    m̄ = dot(z,m)
    m̄inv = 1/m̄

    ρS = dot(ρ̄,m)

    σ3_x = zero(T+first(z)+one(eltype(model)))

    for i ∈ @groups
        x_Si = z[i]*m[i]*m̄inv
        σ3_x += x_Si*x_Si*(_σ[i,i]^3)
        for j ∈ 1:(i-1)
            x_Sj = z[j]*m[j]*m̄inv
            σ3_x += 2*x_Si*x_Sj*(_σ[i,j]^3)
        end
    end
    ρr  = ρS*σ3_x
    
    ϵ = model.vrmodel.params.epsilon
    Tr = T/ϵ[i,j]
    _I = I(model,Tr,ρr)
    
    F = expm1(ϵ_assoc[i,j][a,b]/T)

    return F*K*_I
end

function I(model::SAFTgammaMieModel, Tr,ρr)
    c  = SAFTVRMieconsts.c
    res = zero(ρr+Tr)
    @inbounds for n ∈ 0:10
        ρrn = ρr^n
        res_m = zero(res)
        for m ∈ 0:(10-n)
            res_m += c[n+1,m+1]*Tr^m
        end
        res += res_m*ρrn
    end
    return res
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

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
    λr_s     = params.lambda_r_species
    λa_s     = params.lambda_a_species
    nc_s     = length(nbeads_c)

    ρS_c = eps_v
    g_off = 1
    @inbounds for s in 1:nc_s
        nb_s = nbeads_c[s]
        ρ̄hc_s = 0.0
        @inbounds for kg in g_off:(g_off + nb_s - 1)
            dg = HSd[kg]
            ρ̄hc_s += n[kk, idx_ζ_c, kg] * 3.0/(4.0*_pi*dg^3)
        end
        ρ̄hc_s /= Float64(nb_s)
        ρS_c += ρ̄hc_s * m_s[s]
        g_off += nb_s
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
    g_off = 1
    @inbounds for s in 1:nc_s
        nb_s = nbeads_c[s]
        ρhc_s = 0.0
        @inbounds for kg in g_off:(g_off + nb_s - 1)
            ρhc_s += n[kk, 1, kg]
        end
        ρhc_s /= Float64(nb_s)

        di_s = HSd_s[s]
        λa_c = λa_s[s,s];  λr_c = λr_s[s,s]
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

        g_off += nb_s
    end
    return -res_chain
end

@inline function f_res(out, n, params, T, kk,
                       ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: SAFTgammaMieModel}
    res_hs, = f_hs(n, params.meff, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_disp = f_disp(n, params.meff, params.HSd, params.sigma, params.epsilon,
                      params.lambda_r, params.lambda_a, params.psi_eff,
                      kk, T, Val(NC), Val(ND), Val(6+ND), params.A, params.phi, M)
    res_chain = f_chain(n, params, T, kk, Val(NC), Val(ND), M)
    out[kk] = res_hs + res_chain + res_disp
    return nothing
end

function preallocate_params(system::DFTSystem{<:SAFTgammaMieModel})
    backend = system.options.device
    T_val  = system.structure.conditions[2]
    x_val  = system.structure.ρbulk ./ sum(system.structure.ρbulk)
    HSd_sp = d_gc_av(system.model, 1e-3, T_val, x_val, system.species.size)

    m_vals = system.model.params.segment.values
    S_vals = system.model.params.shapefactor.values
    meff   = m_vals .* S_vals

    params = (;
        HSd               = Adapt.adapt(backend, system.species.size),
        m                 = Adapt.adapt(backend, m_vals),
        S                 = Adapt.adapt(backend, S_vals),
        meff              = Adapt.adapt(backend, meff),
        sigma             = Adapt.adapt(backend, system.model.params.sigma.values),
        epsilon           = Adapt.adapt(backend, system.model.params.epsilon.values),
        lambda_r          = Adapt.adapt(backend, system.model.params.lambda_r.values),
        lambda_a          = Adapt.adapt(backend, system.model.params.lambda_a.values),
        psi_eff           = Adapt.adapt(backend, system.fields[end].width),
        A                 = SAFTVRMIE_A,
        phi               = SAFTVRMIE_PHI,
        nbeads_comp       = Adapt.adapt(backend, system.species.nbeads),
        HSd_species       = Adapt.adapt(backend, HSd_sp),
        m_species         = Adapt.adapt(backend, system.model.vrmodel.params.segment.values),
        sigma_species     = Adapt.adapt(backend, system.model.vrmodel.params.sigma.values),
        epsilon_species   = Adapt.adapt(backend, system.model.vrmodel.params.epsilon.values),
        lambda_r_species  = Adapt.adapt(backend, system.model.vrmodel.params.lambda_r.values),
        lambda_a_species  = Adapt.adapt(backend, system.model.vrmodel.params.lambda_a.values),
    )
    nc = sum(system.species.nbeads)
    return params, nc
end
