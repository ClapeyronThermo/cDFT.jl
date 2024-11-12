function initialize_profiles(model::EoSModel,structure::TwoPhase1DCart, species)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    ρ = zeros(ngrid,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        for j in @chain(i)
            ρ_points = @. tanh_prof(z,ρ1[i],ρ2[i],(bounds[2]/4+3*bounds[1]/4),(2.4728-2.3625*T/Tc)/L)*(z<=(bounds[2]+bounds[1])/2) +
                          tanh_prof(z,ρ2[i],ρ1[i],(3*bounds[2]/4+bounds[1]/4),(2.4728-2.3625*T/Tc)/L)*(z>(bounds[2]+bounds[1])/2)

            ρ[:,j] = ρ_points
        end
    end
    return ρ
end