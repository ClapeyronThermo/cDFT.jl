abstract type PCSAFTFunctionalModel <: SAFTFunctionalModel end

struct PCSAFTFunctional{T <: EoSModel} <: PCSAFTFunctionalModel
    eosmodel::T
    coords::Vector{Float64}
    density::Vector{Float64}
    coords_full::Vector{Float64}
    density_full::Vector{Float64}
    bounds::Float64
end

export PCSAFTFunctional

function PCSAFTFunctional(eosmodel::EoSModel,width::Float64;N::Int64=100)
    σ = maximum(eosmodel.params.sigma.values)
    coords_full = range(-width-2*σ,width+2*σ,length=N)
    coords_full = [coords_full[i] for i in 1:N]
    coords = coords_full[-width .<=coords_full .<=width]
    return PCSAFTFunctional(eosmodel,coords,zeros(length(coords)),coords_full,zeros(N),width)
end

function F(model::PCSAFTFunctionalModel,T)
    return F_hs(model,T)
end

function δFδρ(model::PCSAFTFunctionalModel,T)
    return δFδρ_hs(model,T)
end

export F, δFδρ