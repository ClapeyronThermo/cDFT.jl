using Clapeyron: HeterogcPCPSAFT

struct gcPCPSAFTSpecies <: DFTSpecies
    nbeads::Int64
    bead_id::Vector{Int64}
    connectivity::Array{Int64}
    size::Vector{Float64}
    bulk_density::Float64
    chempot_res::Float64
end

function get_species(model::HeterogcPCPSAFT,structure::DFTStructure)
    (p,T,z) = structure.conditions
    HSd = d(model,1e-3,T,z)
    v = volume(model, p, T, z; phase=:l)
    ρbulk = z./v
    μres = Clapeyron.VT_chemical_potential_res(model, v, T, z) / Clapeyron.R̄ / T

    s = gcPCPSAFTSpecies[]
    for i in @comps
        nbeads = sum(model.groups.n_flattenedgroups[i])        

        _group_id, _group_names, _bond_mat = get_connectivity(model, model.components[i])        
        if isempty(_bond_mat)
            _bond_mat = [0.;;]
        end

        size = zeros(nbeads)
        group_id = zeros(nbeads)
        for j in 1:nbeads
            group_idx = model.groups.groups[i] .== _group_names[j]
            size[j] = HSd[group_idx][1]
            group_id[j] = model.groups.i_groups[i][group_idx][1]
        end

        s = push!(s,gcPCPSAFTSpecies(nbeads, group_id, _bond_mat, size, ρbulk[i], μres[i]))
    end

    return s
end

function get_fields(model::HeterogcPCPSAFT)
    nb = sum(sum(model.groups.n_flattenedgroups))
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
    return f_hs(system,model,n2,n3,n4) + f_hc(system,model,n1,n5) + f_disp(system,model,n6) #+ f_assoc(system,model,n2,n3,n4)
end

function f_hs(system::DFTSystem, model::HeterogcPCPSAFT, n, n₃, nᵥ)
    species = system.species
    m = model.params.segment.values

    n₀ = zero(first(n))
    n₁,n₂,nᵥ₁,nᵥ₂,n₃₃ = zero(n₀), zero(n₀), zero(n₀), zero(n₀), zero(n₀)
    species_id = 1
    bead_id = 1 
    for i in 1:length(n)
        group_id = species[species_id].bead_id[bead_id]
        HSdᵢ = species[species_id].size[bead_id]
        mᵢ,nᵥᵢ = m[group_id],nᵥ[i]
        nᵢmᵢ = n[i]*mᵢ
        n₀ += nᵢmᵢ/HSdᵢ
        n₁ += 0.5nᵢmᵢ
        n₂ += π*nᵢmᵢ*HSdᵢ
        nᵥ₁ += nᵥᵢ*mᵢ/HSdᵢ
        nᵥ₂ += -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[i]*mᵢ

        if bead_id == species[species_id].nbeads
            species_id += 1
            bead_id = 1
        else
            bead_id += 1
        end
    end
    return -n₀*log(1-n₃₃)+(n₁*n₂-nᵥ₂*nᵥ₁)/(1-n₃₃)+(n₂^3/3-n₂*nᵥ₂*nᵥ₂)*(log(1-n₃₃)/(12*π*n₃₃^2)+1/(12*π*n₃₃*(1-n₃₃)^2))
end

function f_hc(system::DFTSystem, model::HeterogcPCPSAFT, ρhc, ρ̄hc)
    species = system.species
    m = model.params.segment.values
    ζ₃ = zero(eltype(ρ̄hc))
    ζ₂ = zero(ζ₃)

    species_id = 1
    bead_id = 1
    for i in 1:length(ρhc)
        group_id = system.species[species_id].bead_id[bead_id]
        HSdi = system.species[species_id].size[bead_id]
        mi,ρ̄hci = m[group_id],ρ̄hc[i]
        ζ₃ += mi*ρ̄hci
        ζ₂ += mi*ρ̄hci/HSdi

        if bead_id == species[species_id].nbeads
            species_id += 1
            bead_id = 1
        else
            bead_id += 1
        end
    end
    ζ₃ *= 0.125
    ζ₂ *= 0.125
    #ζ₃ = 1/8*dot(m,ρ̄hc)
    #ζ₂ = sum(1/8*m.*ρ̄hc./HSd)
    ∑f = zero(ζ₃)
    species_id = 1
    bead_id = 1

    for i in 1:length(ρhc)
        HSd = species[species_id].size
        for j in findall(species[species_id].connectivity[bead_id,:].==1)
            r_HSd = HSd[bead_id]*HSd[j]/(HSd[bead_id]+HSd[j])
            yᵈᵈ = 1/(1-ζ₃) + 3*r_HSd*ζ₂/(1-ζ₃)^2+2*r_HSd^2*ζ₂^2/(1-ζ₃)^3
            fi = -ρhc[i]/2*log(yᵈᵈ)
            ∑f += fi
        end

        if bead_id == species[species_id].nbeads
            species_id += 1
            bead_id = 1
        else
            bead_id += 1
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

    bead_1 = 1
    for i in 1:length(model)
        
        nbead = system.species[i].nbeads

        ρ̄[bead_1:bead_1+nbead-1] .*= 3 ./(4*ψ^3 .*system.species[i].size.^3)./π
        group_id = system.species[i].bead_id
        ρ̄i = sum(ρ̄[bead_1:bead_1+nbead-1])/nbead
        m̄ += sum(m[group_id]*ρ̄i)
        ∑ρ̄i += ρ̄i
        bead_1 += nbead
    end
    m̄ /= ∑ρ̄i

    ∑ρ̄ = sum(ρ̄)

    η = zero(∑ρ̄)
    species_id = 1
    bead_id = 1
    for i in 1:nbeads
        group_id = system.species[species_id].bead_id[bead_id]
        HSd = system.species[species_id].size[bead_id]
        η += m[group_id]*ρ̄[i]*HSd^3

        if bead_id == system.species[species_id].nbeads
            species_id += 1
            bead_id = 1
        else
            bead_id += 1
        end
    end
    η = π/6*η

    C₁ = 1+m̄*(8*η-2*η^2)/(1-η)^4+(1-m̄)*(20*η-27*η^2+12*η^3-2*η^4)/((1-η)^2*(2-η)^2)
    I₁ = I(model,m̄,η,1)
    I₂ = I(model,m̄,η,2)

    m2ϵσ3₂ = zero(T+first(ρ̄))
    m2ϵσ3₁ = m2ϵσ3₂
    
    species_id = 1
    bead_id = 1
    for i in 1:nbeads
        group_idi = system.species[species_id].bead_id[bead_id]
        constant = ρ̄[i]*ρ̄[i]*m[group_idi]*m[group_idi] * σ[group_idi,group_idi]^3
        exp1 = (ϵ[group_idi,group_idi]/T)
        exp2 = exp1*exp1

        m2ϵσ3₁ += constant*exp1
        m2ϵσ3₂ += constant*exp2

        species_id2 = 1
        bead_id2 = 1
        for j in 1:(i-1)
            group_idj = system.species[species_id2].bead_id[bead_id2]
            constant = ρ̄[i]*ρ̄[j]*m[group_idi]*m[group_idj] * σ[group_idi,group_idj]^3
            exp1 = (ϵ[group_idi,group_idj]/T)
            exp2 = exp1*exp1
            m2ϵσ3₁ += 2*constant*exp1
            m2ϵσ3₂ += 2*constant*exp2

            if bead_id2 == system.species[species_id2].nbeads
                species_id2 += 1
                bead_id2 = 1
            else
                bead_id2 += 1
            end
        end

        if bead_id == system.species[species_id].nbeads
            species_id += 1
            bead_id = 1
        else
            bead_id += 1
        end
    end
    return -2*π*I₁*m2ϵσ3₁-π*m̄*C₁^-1*I₂*m2ϵσ3₂
end

function  Δ(model::HeterogcPCPSAFT, T, n, n₃, nᵥ)
    ϵ_assoc = model.params.epsilon_assoc.values
    κ = model.params.bondvol.values
    σ = model.params.sigma.values
    Δout = Compressed4DMatrix{Float64}()
    
    for i in @comps
        k,l = get_group_idx(model,i,j,a,b)
        gkl = @f(g_hs,k,l,_data)
        Δout[idx] = gkl*σ[k,l]^3*(exp(ϵ_assoc[i,j][a,b]/T)-1)*κ[i,j][a,b]
    end
    return Δout
end


function assoc_site_matrix(model::HeterogcPCPSAFT,T,n,n₃,nᵥ,n₀,ξ)
    delta = Δ(model,T,n,n₃,nᵥ)
    sitesparam = Clapeyron.getsites(model)
    _sites = sitesparam.n_sites
    p = _sites.p
    _ii::Vector{Tuple{Int,Int}} = delta.outer_indices
    _aa::Vector{Tuple{Int,Int}} = delta.inner_indices
    _idx = 1:length(_ii)
    _Δ= delta.values
    TT = eltype(_Δ)
    count = 0
    _n = sitesparam.n_sites.v
    nn = length(_n)
    K  = zeros(TT,nn,nn)
    count = 0
    options = assoc_options(model)
    combining = options.combining
    @inbounds for i ∈ 1:length(model) #for i ∈ comps
        sitesᵢ = 1:(p[i+1] - p[i]) #sites are normalized, with independent indices for each component
        for a ∈ sitesᵢ #for a ∈ sites(comps(i))
            ia = compute_index(p,i,a)
            for idx ∈ _idx #iterating for all sites
                ij = _ii[idx]
                ab = _aa[idx]
                if issite(i,a,ij,ab)
                    j = complement_index(i,ij)
                    b = complement_index(a,ab)
                    jb = compute_index(p,j,b)
                    njb = _n[jb]
                    K[ia,jb]  = n₀[j]*ξ[j]*njb*_Δ[idx]
                end
            end
        end
    end
    return K
end

function length_scale(model::HeterogcPCPSAFT)
    return maximum(model.params.sigma.values)
end