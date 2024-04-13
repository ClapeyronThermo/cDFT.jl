Base.length(model::FunctionalModel) = Base.length(model.eosmodel)

function free_energy(system::DFTSystem)
    return F_ideal(system)+F_res(system)
end

onevec(model) = Clapeyron.FillArrays.Ones(length(model))
