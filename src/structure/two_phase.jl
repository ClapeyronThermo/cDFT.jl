function initialize_profiles(model::EoSModel,structure::TwoPhase1DCart, species)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    z = range(first(bounds),last(bounds),ngrid[1]) |> collect
    L = length_scale(model)

    ρ = zeros(ngrid...,sum(species.nbeads))
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

function initialize_profiles(model::EoSModel,structure::TwoPhase2DLamCart, species)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = range(first(bounds[1,:]),last(bounds[1,:]),ngrid[1]) |> collect
    y = range(first(bounds[2,:]),last(bounds[2,:]),ngrid[2]) |> collect

    X = zeros(ngrid[1],ngrid[2])
    Y = zeros(ngrid[1],ngrid[2])

    for i in 1:ngrid[1]
        X[i,:] .= x[i]
    end

    for i in 1:ngrid[2]
        Y[:,i] .= y[i]
    end
    
    L = length_scale(model)

    ρ = zeros(ngrid[1],ngrid[2],sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        for j in @chain(i)
            ρ_points = @. tanh_prof(X,ρ1[i],ρ2[i],(bounds[1,2]/4+3*bounds[1,1]/4),(2.4728-2.3625*T/Tc)/L)*(X<=(bounds[1,2]+bounds[1,1])/2) +
                          tanh_prof(X,ρ2[i],ρ1[i],(3*bounds[1,2]/4+bounds[1,1]/4),(2.4728-2.3625*T/Tc)/L)*(X>(bounds[1,2]+bounds[1,1])/2)
            ρ[:,:,j] = ρ_points
        end
    end
    return ρ
end


function initialize_profiles(model::EoSModel,structure::TwoPhase3DLamCart, species)
    bounds = structure.bounds
    ngrid = structure.ngrid
    nd = length(ngrid)
    (p, T) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = range(first(bounds[1,:]),last(bounds[1,:]),ngrid[1]) |> collect
    X = zeros(ngrid...)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end
    
    L = length_scale(model)

    ρ = zeros(ngrid...,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        for j in @chain(i)
            ρ_points = @. tanh_prof(X,ρ1[i],ρ2[i],(bounds[1,2]/4+3*bounds[1,1]/4),(2.4728-2.3625*T/Tc)/L)*(X<=(bounds[1,2]+bounds[1,1])/2) +
                          tanh_prof(X,ρ2[i],ρ1[i],(3*bounds[1,2]/4+bounds[1,1]/4),(2.4728-2.3625*T/Tc)/L)*(X>(bounds[1,2]+bounds[1,1])/2)
            selectdim(ρ,nd+1,j) .= ρ_points
        end
    end
    return ρ
end