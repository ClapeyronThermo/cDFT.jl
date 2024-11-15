function initialize_profiles(model::EoSModel,structure::TwoPhase1DCart, species)
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    (p, T) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    z = uniform_range(structure) |> collect
    L = length_scale(model)

    ρ = zeros(ngrid...,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        for j in @chain(i)
            coef = (2.4728-2.3625*T/Tc)/L
            ρ_points = @. tanh_prof(z,ρ1[i],ρ2[i],(ub/4+3*lb/4),coef)*(z<=mb) +
                          tanh_prof(z,ρ2[i],ρ1[i],(3*ub/4+lb/4),coef)*(z>mb)

            ρ[:,j] = ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::TwoPhase2DLamCart, species)
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    (p, T) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:] .= x[i]
    end
  
    L = length_scale(model)

    ρ = zeros(ngrid...,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*T/Tc)/L
        for j in @chain(i)
            ρ_points = @. tanh_prof(X,ρ1[i],ρ2[i],(ub/4+3*lb/4),coef)*(X<=mb) +
                          tanh_prof(X,ρ2[i],ρ1[i],(3*ub/4+lb/4),coef)*(X>mb)
            ρ[:,:,j] = ρ_points
        end
    end
    return ρ
end


function initialize_profiles(model::EoSModel,structure::TwoPhase3DLamCart, species)
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = dimension(structure)
    (p, T) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end
    
    L = length_scale(model)

    ρ = zeros(ngrid...,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*T/Tc)/L
        
        for j in @chain(i)
            ρ_points = @. tanh_prof(X,ρ1[i],ρ2[i],(ub/4+3*lb/4),coef)*(X<=mb) +
                          tanh_prof(X,ρ2[i],ρ1[i],(3*ub/4+lb/4),coef)*(X>mb)
            selectdim(ρ,nd+1,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::TwoPhase2DHexCart, species)
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    (p, T) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:] .= x[i]
    end

    y = uniform_range(structure,2)
    Y = zeros(ngrid)

    for i in 1:ngrid[2]
        Y[:,i] .= y[i]
    end

    r = sqrt.(X.^2 + Y.^2)
  
    L = length_scale(model)
    H = ub-lb
    R = H/sqrt(2π)

    ρ = zeros(ngrid...,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*T/Tc)/L
        for j in @chain(i)
            ρ_points = @. tanh_prof(r,ρ1[i],ρ2[i],R,coef)
            ρ[:,:,j] = ρ_points
        end
    end
    return ρ
end