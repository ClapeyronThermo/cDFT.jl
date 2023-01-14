Base.length(model::FunctionalModel) = Base.length(model.eosmodel)

function F_tot(model::EoSModel,ρ,T,z)
    return F_ideal(model.idealmodel,ρ,T,z)+F_res(model,ρ,T,z)
end

onevec(model) = Clapeyron.FillArrays.Ones(length(model))
