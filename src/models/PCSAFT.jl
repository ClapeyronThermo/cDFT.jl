using Clapeyron: PCSAFTModel

"""
    PCSAFT(components::Vector{String})

The PC-SAFT equation of state developed by Gross and Sadowski (2001). Our DFT implementation follows the work of Sauer and Gross (2017) which uses a Weighted Density Functional approach and does not use a chain propagator. The only additional information required in `PCSAFTSpecies` is the bead size at a given temperature.

The bulk model can be obtained from Clapeyron. 
"""
PCSAFT

struct PCSAFTSpecies <: DFTSpecies
    nbeads::Int64
    bead_id::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Float64
    chempot_res::Float64
end

"""
    get_fields(model::EoSModel)

For a given `model`, obtain all of the fields that will be needed to perform the DFT calculation. This function should return a vector of `DFTField`s.
"""
function get_fields(model::PCSAFTModel)
    nc = length(model)
    return [WeightedDensity(:ρ,zeros(nc)),
            WeightedDensity(:∫ρdz,0.5*ones(nc)),
            WeightedDensity(:∫ρz²dz,0.5*ones(nc)),
            WeightedDensity(:∫ρzdz,0.5*ones(nc)),
            WeightedDensity(:∫ρz²dz,ones(nc)),
            WeightedDensity(:∫ρdz,ones(nc)),
            WeightedDensity(:∫ρz²dz,1.3862*ones(nc))]
end

"""
    get_species(model::EoSModel, structure::DFTStructure)

For a given `model` and `structure`, define the relevant parameters for each species. These structs will contain additional information not present by default in the inital `model`, such as the bead size, the number of beads and the connectivity of the beads.
"""
function get_species(model::PCSAFTModel,structure::DFTStructure)
    (p,T,z) = structure.conditions
    size = d(model,1e-3,T,z)
    s = PCSAFTSpecies[]
    v = volume(model, p, T, z; phase=:l)
    ρbulk = z./v
    μres = Clapeyron.VT_chemical_potential_res(model, v, T, z) / Clapeyron.R̄ / T
    for i in @comps
        s = push!(s,PCSAFTSpecies(1, [i], [size[i]], ρbulk[i], μres[i]))
    end
    return s
end

"""
    get_propagator(model::EoSModel)

For a given `model`, define the relevant propagator. 
"""
function get_propagator(model::PCSAFTModel)
    return IdealPropagator()
end


function f_res(system::DFTSystem, model::PCSAFTModel,n)
    n1,n2,n3,n4,n5,n6,n7 = @view(n[1,:]),@view(n[2,:]),@view(n[3,:]),@view(n[4,:]),@view(n[5,:]),@view(n[6,:]),@view(n[7,:])
    return f_hs(system,model,n2,n3,n4) + f_hc(system,model,n1,n5,n6) + f_disp(system,model,n7) + f_assoc(system,model,n2,n3,n4)
end

function f_hc(system::DFTSystem, model::PCSAFTModel, ρhc, ρ̄hc, _λ)
    species = system.species
    HSd = [species[i].size[1] for i in @comps]
    m = model.params.segment.values
    ζ₃ = zero(eltype(HSd)) + zero(eltype(ρ̄hc))
    ζ₂ = zero(ζ₃)
    for i in @comps
        mi,ρ̄hci,HSdi = m[i],ρ̄hc[i],HSd[i]
        ζ₃ += mi*ρ̄hci
        ζ₂ += mi*ρ̄hci/HSdi
    end
    ζ₃ *= 0.125
    ζ₂ *= 0.125
    #ζ₃ = 1/8*dot(m,ρ̄hc)
    #ζ₂ = sum(1/8*m.*ρ̄hc./HSd)
    ∑f = zero(ζ₃)
    for i in @comps
        λ = _λ[i]/(2*HSd[i])
        yᵈᵈ = 1/(1-ζ₃) + 1.5*HSd[i]*ζ₂/(1-ζ₃)^2+0.5*HSd[i]^2*ζ₂^2/(1-ζ₃)^3
        fi = -ρhc[i]*(m[i]-1)*log(yᵈᵈ*λ/ρhc[i])
        ∑f += fi
    end
    return ∑f
end

function f_disp(system::DFTSystem, model::PCSAFTModel, ρ̄)
    species = system.species
    (_, T, _) = system.structure.conditions
    ψ = 1.3862
    σ = model.params.sigma.values
    m = model.params.segment.values
    HSd = [species[i].size[1] for i in @comps]

    ρ̄ = ρ̄*3 ./(4*ψ^3 .*HSd.^3)/π
    ∑ρ̄ = sum(ρ̄)
    x = ρ̄ /∑ρ̄
    m̄ = dot(x,m)

    η = zero(first(m) + ∑ρ̄ + first(HSd))
    for i in 1:length(m)
        η += m[i]*ρ̄[i]*HSd[i]^3
    end
    η = π/6*η

    C₁ = 1+m̄*(8*η-2*η^2)/(1-η)^4+(1-m̄)*(20*η-27*η^2+12*η^3-2*η^4)/((1-η)^2*(2-η)^2)
    I₁ = I(model,m̄,η,1)
    I₂ = I(model,m̄,η,2)

    m2ϵσ3₁,m2ϵσ3₂ =  Clapeyron.m2ϵσ3(model,zero(T), T, x)
    return -2*π*∑ρ̄^2*I₁*m2ϵσ3₁-π*∑ρ̄^2*m̄*C₁^-1*I₂*m2ϵσ3₂
end

function I(model::PCSAFTModel,m̄,n₃,n)
    if n == 1
        corr = Clapeyron.PCSAFTconsts.corr1
    elseif n == 2
        corr = Clapeyron.PCSAFTconsts.corr2
    end
    res = zero(n₃)
    @inbounds for i ∈ 1:7
        ii = i-1
        corr1,corr2,corr3 = corr[i]
        ki = corr1 + (m̄-1)/m̄*corr2 + (m̄-1)/m̄*(m̄-2)/m̄*corr3
        res += ki*n₃^ii
    end
    return res
end

function Δ(model::PCSAFTModel, T, n, n₃, nᵥ, i, j, a, b)
    ϵ_assoc = model.params.epsilon_assoc.values
    κ = model.params.bondvol.values
    κijab = κ[i,j][a,b]
    _0 = zero(T+first(n)+first(n₃)+first(nᵥ)+first(κijab))
    iszero(κijab) && return _0

    σ = model.params.sigma.values[i,j]
    m = model.params.segment.values
    HSd = d(model,1e-3,T,onevec(model))
    dij = (HSd[i]*HSd[j])/(HSd[i]+HSd[j])

    n₂, nᵥ₂, n₃₃ = _0,_0,_0
    for i in 1:length(n)
        nᵢ,mᵢ,nᵥᵢ,HSdᵢ = n[i],m[i],nᵥ[i],HSd[i]
        n₂ += π*HSdᵢ*nᵢ*mᵢ
        nᵥ₂ += -2π*nᵥᵢ*mᵢ
        n₃₃ += n₃[i]*mᵢ
    end
    #n₂ = sum(π.*HSd.*n.*m)
    #nᵥ₂ = sum(-2π.*nᵥ.*m)
    #n₃  = sum(n₃.*m)

    ξ = 1-nᵥ₂^2/n₂^2
    g_hs = 1/(1-n₃₃)+dij*ξ*n₂/(2*(1-n₃₃)^2)+dij^2*n₂^2*ξ/(18*(1-n₃₃)^3)
    return g_hs*σ^3*expm1(ϵ_assoc[i,j][a,b]/T)*κijab
end

"""
    length_scale(model::EoSModel)

Obtains the maximum length scale in the model and helps define the dimensions of the DFT system. This is typically equal to the size of the largest bead.
"""
function length_scale(model::SAFTModel)
    return maximum(model.params.sigma.values)
end

export length_scale