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

function PCSAFTFunctional(eosmodel::EoSModel,width::Float64;dz::Float64=0.01)
    σ = maximum(eosmodel.params.sigma.values)
    N = Int(2*(width+2)/dz)+1
    coords_full = range(-width-2,width+2,length=N)
    coords_full = [coords_full[i] for i in 1:N]
    coords = coords_full[-width .<=coords_full .<=width]
    return PCSAFTFunctional(eosmodel,coords,zeros(length(coords)),coords_full,zeros(N),width)
end

function F(model::PCSAFTFunctionalModel,T)
    m = model.eosmodel.params.segment.values[1]
    return m*F_hs(model,T)+F_hc(model,T)
end

function δFδρ(model::PCSAFTFunctionalModel,T)
    return δFδρ_hs(model,T)+δFδρ_hc(model,T)
end

export F, δFδρ