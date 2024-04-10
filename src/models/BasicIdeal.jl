using Clapeyron: BasicIdealModel

function F_ideal(model::BasicIdealModel,ρ,T,z)
    dz = ρ[1].mesh_size

    ρ_eval = zeros(length(z),length(ρ))
    for i in 1:length(ρ)
        ρ_eval[:,i] = ρ[i].density
    end
    nc = length(ρ)
    idx = 1:nc

    f0(x) = f_ideal(model,T,x[idx])
    Φ = mapslices(f0,ρ_eval;dims=2)
    return ∫(Φ,dz)
end

function f_ideal(model::BasicIdealModel, T, ρ)
    ∑f = zero(T + first(ρ))
    lnT = 1.5*log(T)
    for i in 1:length(ρ)
        ρᵢ = ρ[i]
        ∑f += N_A*ρᵢ*(log(ρᵢ)  - lnT - 1)
    end
    return ∑f
end