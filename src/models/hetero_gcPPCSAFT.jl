using Clapeyron: HeterogcPCPSAFT

struct gcPCPSAFTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    connectivity::Array{Int64}
    species_id::Vector{Int64}
    group_id::Vector{Int64}
    size::Vector{Float64}
end

function get_species(model::HeterogcPCPSAFT,structure::DFTStructure)
    (p,T,z) = structure.conditions
    nc = length(model)
    nbeads = sum.(model.groups.n_flattenedgroups)
    HSd = d(model,1e-3,T,z)
    species_id = zeros(Int64, sum(nbeads))
    group_id = zeros(Int64, sum(nbeads))
    size = zeros(sum(nbeads))
    connectivity = zeros(Int64, sum(nbeads), sum(nbeads))
    for i in 1:nc
        species_id[sum(nbeads[1:i])-nbeads[i]+1:sum(nbeads[1:i])] .= i
        _group_id, _bond_mat = get_connectivity(model, model.components[i])
        group_id[sum(nbeads[1:i])-nbeads[i]+1:sum(nbeads[1:i])] = _group_id
        if !isempty(_bond_mat)
            connectivity[sum(nbeads[1:i])-nbeads[i]+1:sum(nbeads[1:i]),sum(nbeads[1:i])-nbeads[i]+1:sum(nbeads[1:i])] = _bond_mat
        end
    end

    for i in unique(group_id)
        size[group_id .== i] .= HSd[i]
    end

    return gcPCPSAFTSpecies(nbeads, connectivity, species_id, group_id, size)
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

function f_res(system::DFTSystem, model::HeterogcPCPSAFT,n)
    n1,n2,n3,n4,n5,n6 = @view(n[1,:]),@view(n[2,:]),@view(n[3,:]),@view(n[4,:]),@view(n[5,:]),@view(n[6,:])
    return f_hs(system,model,n2,n3,n4) + f_hc(system,model,n1,n5) + f_disp(system,model,n6) #+ f_assoc(system,model,n2,n3,n4)
end

function f_hs(system::DFTSystem, model::HeterogcPCPSAFT, n, n₃, nᵥ)
    species = system.species
    m = model.params.segment.values
    HSd = system.species.size

    n₀ = zero(first(n) + first(m) + first(HSd))
    n₁,n₂,nᵥ₁,nᵥ₂,n₃₃ = zero(n₀), zero(n₀), zero(n₀), zero(n₀), zero(n₀)
    for i in 1:length(n)
        group_id = species.group_id[i]
        mᵢ,HSdᵢ,nᵥᵢ = m[group_id],HSd[i],nᵥ[i]
        nᵢmᵢ = n[i]*mᵢ
        n₀ += nᵢmᵢ/HSdᵢ
        n₁ += 0.5nᵢmᵢ
        n₂ += π*nᵢmᵢ*HSdᵢ
        nᵥ₁ += nᵥᵢ*mᵢ/HSdᵢ
        nᵥ₂ += -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[i]*mᵢ
    end
    return -n₀*log(1-n₃₃)+(n₁*n₂-nᵥ₂*nᵥ₁)/(1-n₃₃)+(n₂^3/3-n₂*nᵥ₂*nᵥ₂)*(log(1-n₃₃)/(12*π*n₃₃^2)+1/(12*π*n₃₃*(1-n₃₃)^2))
end

function f_hc(system::DFTSystem, model::HeterogcPCPSAFT, ρhc, ρ̄hc)
    HSd = system.species.size
    nbeads = sum(system.species.nbeads)
    connectivity = system.species.connectivity
    m = model.params.segment.values
    ζ₃ = zero(eltype(HSd)) + zero(eltype(ρ̄hc))
    ζ₂ = zero(ζ₃)
    for i in 1:nbeads
        group_id = system.species.group_id[i]
        mi,ρ̄hci,HSdi = m[group_id],ρ̄hc[i],HSd[i]
        ζ₃ += mi*ρ̄hci
        ζ₂ += mi*ρ̄hci/HSdi
    end
    ζ₃ *= 0.125
    ζ₂ *= 0.125
    #ζ₃ = 1/8*dot(m,ρ̄hc)
    #ζ₂ = sum(1/8*m.*ρ̄hc./HSd)
    ∑f = zero(ζ₃)
    for i in 1:nbeads
        for j in findall(connectivity[i,:].==1)
            r_HSd = HSd[i]*HSd[j]/(HSd[i]+HSd[j])
            yᵈᵈ = 1/(1-ζ₃) + 3*r_HSd*ζ₂/(1-ζ₃)^2+2*r_HSd^2*ζ₂^2/(1-ζ₃)^3
            fi = -ρhc[i]/2*log(yᵈᵈ)
            ∑f += fi
        end
    end
    return ∑f
end

function f_disp(system::DFTSystem, model::HeterogcPCPSAFT, ρ̄)
    HSd = system.species.size
    nbeads = sum(system.species.nbeads)
    (_, T, _) = system.structure.conditions
    ψ = 1.5357
    σ = model.params.sigma.values
    ϵ = model.params.epsilon.values
    m = model.params.segment.values

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
    ∑ρ̄ = sum(ρ̄)
    m̄ = zero(∑ρ̄)
    ∑ρ̄i = zero(∑ρ̄)

    for i in 1:length(model)
        species_id = system.species.species_id
        group_id = system.species.group_id
        ρ̄i = sum(ρ̄[species_id .== i])/system.species.nbeads[i]
        m̄ += sum(m[group_id[species_id .== i]]*ρ̄i)
        ∑ρ̄i += ρ̄i
    end
    m̄ /= ∑ρ̄i


    η = zero( ∑ρ̄ + first(HSd))
    for i in 1:nbeads
        group_id = system.species.group_id[i]
        η += m[group_id]*ρ̄[i]*HSd[i]^3
    end
    η = π/6*η

    C₁ = 1+m̄*(8*η-2*η^2)/(1-η)^4+(1-m̄)*(20*η-27*η^2+12*η^3-2*η^4)/((1-η)^2*(2-η)^2)
    I₁ = I(model,m̄,η,1)
    I₂ = I(model,m̄,η,2)

    m2ϵσ3₂ = zero(T+first(ρ̄))
    m2ϵσ3₁ = m2ϵσ3₂
    
    for i in 1:nbeads
        group_idi = system.species.group_id[i]
        constant = ρ̄[i]*ρ̄[i]*m[group_idi]*m[group_idi] * σ[group_idi,group_idi]^3
        exp1 = (ϵ[group_idi,group_idi]/T)
        exp2 = exp1*exp1

        m2ϵσ3₁ += constant*exp1
        m2ϵσ3₂ += constant*exp2

        for j in 1:(i-1)
            group_idj = system.species.group_id[j]
            constant = ρ̄[i]*ρ̄[j]*m[group_idi]*m[group_idj] * σ[group_idi,group_idj]^3
            exp1 = (ϵ[group_idi,group_idj]/T)
            exp2 = exp1*exp1
            m2ϵσ3₁ += 2*constant*exp1
            m2ϵσ3₂ += 2*constant*exp2
        end
    end
    return -2*π*I₁*m2ϵσ3₁-π*m̄*C₁^-1*I₂*m2ϵσ3₂
end

function length_scale(model::HeterogcPCPSAFT)
    return maximum(model.params.sigma.values)
end