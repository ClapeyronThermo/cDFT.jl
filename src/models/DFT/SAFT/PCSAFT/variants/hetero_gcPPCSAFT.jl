using Clapeyron: HeterogcPCPSAFT, pcp_segment, pcp_sigma, pcp_epsilon, pcp_dipole2

function DFTSystem(model::HeterogcPCPSAFT,structure::DFTStructure,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure, options.device)
    propagator = get_propagator(model, species, structure, options.device)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, fields, nothing, propagator, options, chunksize)
end

function DFTSystem(model::HeterogcPCPSAFT,structure::DFTStructure, external_fields,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model, species, structure, options.device)
    propagator = get_propagator(model, species, structure, options.device)
    NF = compute_field_len(fields,dimension(structure))
    chunksize = Val{NF}()
    return DFTSystem(model, species, structure, fields, external_fields, propagator, options, chunksize)
end

struct gcPCPSAFTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::HeterogcPCPSAFT,structure::DFTStructure)
    (p,T) = structure.conditions
    z = structure.ρbulk
    v = 1/sum(z)

    HSd = d(model,1e-3,T,z)
    μres = Clapeyron.VT_chemical_potential_res(model, v, T, z/sum(z)) / Clapeyron.R̄ / T
    nbeads = length.(model.groups.groups)

    levels = zeros(Int, sum(nbeads))

    for i in @comps
        i_groups = model.groups.i_groups[i]
        bond_mat = Int.(model.groups.n_intergroups[i]) .> 0
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
    return gcPCPSAFTSpecies(nbeads,HSd,levels,structure.ρbulk,μres)
end

function get_fields(model::HeterogcPCPSAFT, species::DFTSpecies, structure::DFTStructure, backend::Backend)
    nb = sum(species.nbeads)
    ngrid = structure.ngrid
    ω = structure_ω(structure, backend)
    d = species.size
    ψ = 1.5357
    return [SWeightedDensity(:ρ,zeros(nb),ω,ngrid,backend),
            SWeightedDensity(:∫ρdz,0.5*d,ω,ngrid,backend),
            SWeightedDensity(:∫ρz²dz,0.5*d,ω,ngrid,backend),
            VWeightedDensity(:∫ρzdz,0.5*d,ω,ngrid,backend),
            SWeightedDensity(:∫ρz²dz,d,ω,ngrid,backend),
            SWeightedDensity(:∫ρz²dz,d .* ψ,ω,ngrid,backend)]

end

function get_propagator(model::HeterogcPCPSAFT, species::DFTSpecies, structure::DFTStructure, backend::Backend)
    return TangentHSPropagator(model, species, structure, backend)
end

function f_res(system::DFTSystem, model::HeterogcPCPSAFT,n)
    nd = dimension(system)
    n1,n2,n3,n4,n5,n6 = @view(n[1,:]),@view(n[2,:]),@view(n[3,:]),@view(n[4:4+nd-1,:]),@view(n[4+nd,:]),@view(n[5+nd,:])
    return f_hs(system,model,n2,n3,n4) + f_hc(system,model,n1,n5) + f_disp(system,model,n6) + f_assoc(system,model,n2,n3,n4) + f_polar(system,model,n6)
end

function f_hs(system::DFTSystem, model::HeterogcPCPSAFT, n, n₃, nᵥ)
    species = system.species
    m = model.params.segment.values

    n₀ = zero(first(n))
    n₁,n₂,nᵥ₁,nᵥ₂,n₃₃ = zero(n₀), zero(n₀), zero(nᵥ[:,1]), zero(nᵥ[:,1]), zero(n₀)
    for i in @comps
        for k in @groups(i)
            HSdᵢ = species.size[k]
            mᵢ,nᵥᵢ = m[k],nᵥ[:,k]
            nᵢmᵢ = n[k]*mᵢ
            n₀ += nᵢmᵢ/HSdᵢ
            n₁ += 0.5nᵢmᵢ
            n₂ += π*nᵢmᵢ*HSdᵢ
            nᵥ₁ .+= nᵥᵢ*mᵢ/HSdᵢ
            nᵥ₂ .+= -2π*nᵥᵢ*mᵢ
            n₃₃ += n₃[k]*mᵢ
        end
    end
    nᵥ₁nᵥ₂ = dot(nᵥ₁,nᵥ₂)
    nᵥ₂nᵥ₂ = dot(nᵥ₂,nᵥ₂)
    return -n₀*log(1-n₃₃)+(n₁*n₂-nᵥ₁nᵥ₂)/(1-n₃₃)+(n₂^3/3-n₂*nᵥ₂nᵥ₂)*(log(1-n₃₃)/(12*π*n₃₃^2)+1/(12*π*n₃₃*(1-n₃₃)^2))
end

function f_hc(system::DFTSystem, model::HeterogcPCPSAFT, ρhc, ρ̄hc)
    species = system.species
    m = model.params.segment.values
    ζ₃ = zero(eltype(ρ̄hc))
    ζ₂ = zero(ζ₃)

    for i in @comps
        for k in @groups(i)
            HSdi = species.size[k]
            mi,ρ̄hci = m[k],ρ̄hc[k]
            ζ₃ += mi*ρ̄hci
            ζ₂ += mi*ρ̄hci/HSdi
        end
    end
    ζ₃ *= 0.125
    ζ₂ *= 0.125
    #ζ₃ = 1/8*dot(m,ρ̄hc)
    #ζ₂ = sum(1/8*m.*ρ̄hc./HSd)
    ∑f = zero(ζ₃)

    for i in @comps
        n_intergroups = model.groups.n_intergroups[i]
        HSd = species.size
        for k in @groups(i)
            for l in findall(n_intergroups[k,:].==1)
                r_HSd = HSd[k]*HSd[l]/(HSd[k]+HSd[l])
                yᵈᵈ = 1/(1-ζ₃) + 3*r_HSd*ζ₂/(1-ζ₃)^2+2*r_HSd^2*ζ₂^2/(1-ζ₃)^3
                fi = -ρhc[k]/2*log(yᵈᵈ)
                ∑f += fi
            end
        end
    end
    return ∑f
end

function f_disp(system::DFTSystem, model::HeterogcPCPSAFT, n)
    ρ̄ = deepcopy(n)
    nbeads = length(ρ̄)
    T = system.structure.conditions[2]
    ψ = 1.5357
    σ = model.params.sigma.values
    ϵ = model.params.epsilon.values
    m = model.params.segment.values
    
    m̄ = zero(first(ρ̄))
    ∑ρ̄i = zero(first(ρ̄))
    η = zero(first(ρ̄))

    for i in @comps
        for k in @groups(i)
            ρ̄[k] *= 3 /(4*ψ^3 *system.species.size[k].^3)/π
            m̄ += m[k]*ρ̄[k]
            η += m[k]*ρ̄[k]*system.species.size[k]^3
            ∑ρ̄i += ρ̄[k]/system.species.nbeads[i]
        end
    end
    m̄ /= ∑ρ̄i
    η *= π/6

    C₁ = 1+m̄*(8*η-2*η^2)/(1-η)^4+(1-m̄)*(20*η-27*η^2+12*η^3-2*η^4)/((1-η)^2*(2-η)^2)
    I₁ = I(model,m̄,η,1)
    I₂ = I(model,m̄,η,2)

    m2ϵσ3₂ = zero(T+first(ρ̄))
    m2ϵσ3₁ = m2ϵσ3₂
    
    for i in 1:nbeads
        constant = ρ̄[i]*ρ̄[i]*m[i]*m[i] * σ[i,i]^3
        exp1 = (ϵ[i,i]/T)
        exp2 = exp1*exp1

        m2ϵσ3₁ += constant*exp1
        m2ϵσ3₂ += constant*exp2
        for j in 1:(i-1)
            constant = ρ̄[i]*ρ̄[j]*m[i]*m[j] * σ[i,j]^3
            exp1 = (ϵ[i,j]/T)
            exp2 = exp1*exp1
            m2ϵσ3₁ += 2*constant*exp1
            m2ϵσ3₂ += 2*constant*exp2
        end
    end
    return -2*π*I₁*m2ϵσ3₁-π*m̄*C₁^-1*I₂*m2ϵσ3₂
end


# function  Δ(model::HeterogcPCPSAFT, T, n, n₃, nᵥ)
#     ϵ_assoc = model.params.epsilon_assoc.values
#     κ = model.params.bondvol.values
#     σ = model.params.sigma.values
#     Δout = 
    
#     for (idx,(i,j),(a,b)) in indices(κ)
#         k,l = get_chain_idx(model,i,j,a,b)
#         gkl = @f(g_hs,k,l,_data)
#         Δout[idx] = gkl*σ[k,l]^3*(exp(ϵ_assoc[i,j][a,b]/T)-1)*κ[i,j][a,b]
#     end
#     return Δout
# end

function Δ(model::HeterogcPCPSAFT, T, n, n₃, nᵥ, i, j, a, b)
    ϵ_assoc = model.params.epsilon_assoc.values
    κ = model.params.bondvol.values
    κijab = κ[i,j][a,b]
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(κijab))
    iszero(κijab) && return _0

    k,l = get_chain_idx(model,i,j,a,b)
    σ = model.params.sigma.values[k,l]
    m = model.params.segment.values
    HSd = d(model,1e-3,T,onevec(model))
    dij = (HSd[k]*HSd[l])/(HSd[k]+HSd[l])

    n₂, nᵥ₂, n₃₃ = _0,zero(nᵥ[:,i]),_0
    for kk in 1:length(n)
        nᵢ,mᵢ,nᵥᵢ,HSdᵢ = n[kk],m[kk],nᵥ[:,kk],HSd[kk]
        n₂ += π*HSdᵢ*nᵢ*mᵢ
        nᵥ₂ .+= -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[kk]*mᵢ
    end
    nᵥ₂nᵥ₂ = dot(nᵥ₂,nᵥ₂)

    #n₂ = sum(π.*HSd.*n.*m)
    #nᵥ₂ = sum(-2π.*nᵥ.*m)
    #n₃  = sum(n₃.*m)

    ξ = 1-nᵥ₂nᵥ₂/n₂^2
    g_hs = 1/(1-n₃₃)+dij*ξ*n₂/(2*(1-n₃₃)^2)+dij^2*n₂^2*ξ/(18*(1-n₃₃)^3)
    return g_hs*σ^3*expm1(ϵ_assoc[i,j][a,b]/T)*κijab
end

function f_polar(system::DFTSystem, model::HeterogcPCPSAFT, ρ̄)
    species = system.species
    T = system.structure.conditions[2]
    μ̄² = pcp_dipole2(model)
    has_dp = !all(iszero, μ̄²)
    if !has_dp return zero(T+first(ρ̄)) end

    ψ = 1.5357
    HSd = species.size

    m = pcp_segment(model)
    ϵ = pcp_epsilon(model)
    σ = pcp_sigma(model)

    _m = model.params.segment.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π

    _ρ̄ = zeros(eltype(ρ̄),length(model))
    η = zero(first(ρ̄))
    for i in @comps
        for k in @groups(i)
            _ρ̄[i] += ρ̄[k]/system.species.nbeads[i]
            η += _m[k]*ρ̄[k]*system.species.size[k]^3
        end
    end

    η *= π/6
    ∑ρ̄ = sum(_ρ̄)
    x = _ρ̄ /∑ρ̄
    _A₂ = A2(x,m,ϵ,σ,μ̄²,η,∑ρ̄,T)
    iszero(_A₂) && return zero(_A₂)
    _A₃ = A3(x,m,ϵ,σ,μ̄²,η,∑ρ̄,T)
    _a_dd = _A₂^2/(_A₂-_A₃)
    return ∑ρ̄*_a_dd
end

function length_scale(model::HeterogcPCPSAFT)
    return maximum(model.params.sigma.values)
end

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for HeterogcPCPSAFT:
FMT hard-sphere + hard-chain bonding (Wertheim topology) + PCSAFT-style dispersion.
Polar term is NOT included in the Enzyme kernel (requires per-component aggregation).
Chain connectivity is handled by TangentHSPropagator.

Field layout (6 fields total):
  1        : ρ (unweighted)
  2        : ∫ρdz  with 0.5*d → n₀, n₁, n₂
  3        : ∫ρz²dz with 0.5*d → n₃
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ
  4+ND     : ∫ρz²dz with d    → ρ̄hc (bonding)
  5+ND     : ∫ρz²dz with d*ψ → ρ̄z  (dispersion, ψ=1.5357)

NC = total number of groups (sum of nbeads per component).
"""
@inline function f_hc(n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: HeterogcPCPSAFT}
    _pi   = 3.141592653589793
    eps_v = 1e-15
    HSd   = params.HSd
    m_seg = params.m
    n_bonds = params.n_bonds
    bond_k  = params.bond_k
    bond_l  = params.bond_l

    idx_ζ = 4 + ND
    ζ₃ = 0.0; ζ₂ = 0.0
    @inbounds for i in 1:NC
        mi = m_seg[i]; di = HSd[i]; ρ̄hci = n[kk, idx_ζ, i]
        ζ₃ += mi * ρ̄hci
        ζ₂ += mi * ρ̄hci / di
    end
    ζ₃ *= 0.125; ζ₂ *= 0.125
    inv1ζ₃ = 1.0 / (1.0 - ζ₃ + eps_v)

    res_hc = 0.0
    @inbounds for ib in 1:n_bonds
        k = bond_k[ib]; l = bond_l[ib]
        dk = HSd[k]; dl = HSd[l]
        r_HSd = dk * dl / (dk + dl)
        ζ₂_ov3 = ζ₂ * inv1ζ₃
        yᵈᵈ = inv1ζ₃ + 3.0*r_HSd*ζ₂_ov3*inv1ζ₃ + 2.0*r_HSd^2*ζ₂_ov3^2*inv1ζ₃
        ρhck = n[kk, 1, k]
        res_hc += -ρhck * 0.5 * Base.log(abs(yᵈᵈ) + eps_v)
    end
    return res_hc
end

@inline function f_disp(n, params, T, kk, ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: HeterogcPCPSAFT}
    _pi   = 3.141592653589793
    eps_v = 1e-15
    HSd              = params.HSd
    m_seg            = params.m
    σ                = params.sigma
    ϵ                = params.epsilon
    nbeads_for_group = params.nbeads_for_group

    ψ       = 1.5357
    idx_ρz  = 5 + ND
    factor  = 3.0 / (4.0*ψ*ψ*ψ*_pi)
    ρ̄_tot   = eps_v; m̄_num = 0.0; η_sum = 0.0
    @inbounds for i in 1:NC
        di  = HSd[i]
        ρ̄i  = n[kk, idx_ρz, i] * factor / (di*di*di)
        m̄_num += m_seg[i] * ρ̄i
        η_sum  += m_seg[i] * ρ̄i * di*di*di
        ρ̄_tot  += ρ̄i / nbeads_for_group[i]
    end
    m̄  = m̄_num / ρ̄_tot
    ηd = _pi/6.0 * η_sum

    m2ϵσ3_1 = 0.0; m2ϵσ3_2 = 0.0
    @inbounds for i in 1:NC
        di   = HSd[i]
        ρ̄i   = n[kk, idx_ρz, i] * factor / (di*di*di)
        @inbounds for j in i:NC
            dj   = HSd[j]
            ρ̄j   = n[kk, idx_ρz, j] * factor / (dj*dj*dj)
            cij  = ρ̄i * ρ̄j * m_seg[i] * m_seg[j] * σ[i,j]*σ[i,j]*σ[i,j]
            eT   = ϵ[i,j] / (T + eps_v)
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
    return -2.0*_pi*I₁*m2ϵσ3_1 - _pi*m̄*I₂*m2ϵσ3_2 / (C₁ + eps_v)
end

@inline function f_res(out, n, params, T, kk,
                       ::Val{NC}, ::Val{ND}, ::Type{M}) where {NC, ND, M <: HeterogcPCPSAFT}
    res_hs, = f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_hc  = f_hc(n, params, T, kk, Val(NC), Val(ND), M)
    res_disp = f_disp(n, params, T, kk, Val(NC), Val(ND), M)
    out[kk] = res_hs + res_hc + res_disp
    return nothing
end

function preallocate_params(system::DFTSystem{<:HeterogcPCPSAFT})
    backend  = system.options.device
    model    = system.model
    nc_spec  = length(model)
    nbeads   = system.species.nbeads
    nc_groups = sum(nbeads)

    bond_k_list = Int32[]
    bond_l_list = Int32[]
    for i in 1:nc_spec
        i_groups        = model.groups.i_groups[i]
        n_intergroups_i = model.groups.n_intergroups[i]
        for k in i_groups
            for l in findall(n_intergroups_i[k,:] .== 1)
                push!(bond_k_list, Int32(k))
                push!(bond_l_list, Int32(l))
            end
        end
    end

    nbeads_for_group = Vector{Float64}(undef, nc_groups)
    for i in 1:nc_spec
        nbi = Float64(nbeads[i])
        for k in model.groups.i_groups[i]
            nbeads_for_group[k] = nbi
        end
    end

    params = (;
        HSd              = Adapt.adapt(backend, system.species.size),
        m                = Adapt.adapt(backend, system.model.params.segment.values),
        sigma            = Adapt.adapt(backend, system.model.params.sigma.values),
        epsilon          = Adapt.adapt(backend, system.model.params.epsilon.values),
        nbeads_for_group = Adapt.adapt(backend, nbeads_for_group),
        n_bonds          = length(bond_k_list),
        bond_k           = Adapt.adapt(backend, bond_k_list),
        bond_l           = Adapt.adapt(backend, bond_l_list),
    )
    nc = nc_groups
    return params, nc
end