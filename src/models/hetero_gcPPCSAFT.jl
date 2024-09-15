using Clapeyron: HeterogcPCPSAFT

function DFTSystem(model::HeterogcPCPSAFT,structure::DFTStructure,options::DFTOptions)
    model = expand_model(model)
    species = get_species(model, structure)
    fields = get_fields(model)
    propagator = get_propagator(model)
    profiles = initialize_profiles(model,structure, species)
    group_idx = reduce(vcat,model.groups.i_groups)
    profiles[group_idx] = profiles
    return DFTSystem(model, species, structure, profiles, fields, propagator, options)
end

struct gcPCPSAFTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function get_species(model::HeterogcPCPSAFT,structure::DFTStructure)
    (p,T,z) = structure.conditions
    HSd = d(model,1e-3,T,z)
    v = volume(model, p, T, z; phase=:l)
    ρbulk = z./v
    μres = Clapeyron.VT_chemical_potential_res(model, v, T, z) / Clapeyron.R̄ / T
    nbeads = length.(model.groups.groups)

    levels = zeros(Int, sum(nbeads))

    for i in @comps
        i_groups = model.groups.i_groups[i]
        bond_mat = Bool.(model.groups.n_intergroups[i])
        nbonds = sum(bond_mat,dims=2)[:]
        is_leaf = nbonds .== 1
        i_root = findfirst(nbonds .== maximum(nbonds[i_groups]))
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
    return gcPCPSAFTSpecies(nbeads,HSd,levels,ρbulk,μres)
end

function get_fields(model::HeterogcPCPSAFT)
    nb = length(model.groups.flattenedgroups)
    return [WeightedDensity(:ρ,zeros(nb)),
            WeightedDensity(:∫ρdz,0.5*ones(nb)),
            WeightedDensity(:∫ρz²dz,0.5*ones(nb)),
            WeightedDensity(:∫ρzdz,0.5*ones(nb)),
            WeightedDensity(:∫ρz²dz,ones(nb)),
            WeightedDensity(:∫ρz²dz,1.5357*ones(nb))]
end

function get_propagator(model::HeterogcPCPSAFT)
    return TangentHSPropagator()
end

function f_res(system::DFTSystem, model::HeterogcPCPSAFT,n)
    n1,n2,n3,n4,n5,n6 = @view(n[1,:]),@view(n[2,:]),@view(n[3,:]),@view(n[4,:]),@view(n[5,:]),@view(n[6,:])
    return f_hs(system,model,n2,n3,n4) + f_hc(system,model,n1,n5) + f_disp(system,model,n6) + f_assoc(system,model,n2,n3,n4)
end

function f_hs(system::DFTSystem, model::HeterogcPCPSAFT, n, n₃, nᵥ)
    species = system.species
    m = model.params.segment.values

    n₀ = zero(first(n))
    n₁,n₂,nᵥ₁,nᵥ₂,n₃₃ = zero(n₀), zero(n₀), zero(n₀), zero(n₀), zero(n₀)
    for i in @comps
        for k in @groups(i)
            HSdᵢ = species.size[k]
            mᵢ,nᵥᵢ = m[k],nᵥ[k]
            nᵢmᵢ = n[k]*mᵢ
            n₀ += nᵢmᵢ/HSdᵢ
            n₁ += 0.5nᵢmᵢ
            n₂ += π*nᵢmᵢ*HSdᵢ
            nᵥ₁ += nᵥᵢ*mᵢ/HSdᵢ
            nᵥ₂ += -2π*nᵥᵢ*mᵢ
            n₃₃ += n₃[k]*mᵢ
        end
    end
    return -n₀*log(1-n₃₃)+(n₁*n₂-nᵥ₂*nᵥ₁)/(1-n₃₃)+(n₂^3/3-n₂*nᵥ₂*nᵥ₂)*(log(1-n₃₃)/(12*π*n₃₃^2)+1/(12*π*n₃₃*(1-n₃₃)^2))
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
    (_, T, _) = system.structure.conditions
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

    n₂, nᵥ₂, n₃₃ = _0,_0,_0
    for k in 1:length(n)
        nᵢ,mᵢ,nᵥᵢ,HSdᵢ = n[k],m[k],nᵥ[k],HSd[k]
        n₂ += π*HSdᵢ*nᵢ*mᵢ
        nᵥ₂ += -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[k]*mᵢ
    end
    #n₂ = sum(π.*HSd.*n.*m)
    #nᵥ₂ = sum(-2π.*nᵥ.*m)
    #n₃  = sum(n₃.*m)

    ξ = 1-nᵥ₂^2/n₂^2
    g_hs = 1/(1-n₃₃)+dij*ξ*n₂/(2*(1-n₃₃)^2)+dij^2*n₂^2*ξ/(18*(1-n₃₃)^3)
    return g_hs*σ^3*expm1(ϵ_assoc[i,j][a,b]/T)*κijab
end

function length_scale(model::HeterogcPCPSAFT)
    return maximum(model.params.sigma.values)
end