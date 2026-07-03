function initialize_profiles(model::EoSModel,structure::TwoPhase1DCart, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub-lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    z = uniform_range(structure) |> collect
    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*temperature/Tc)*H/L
        coef = sqrt(coef^2-1)/4
        for j in @chain(i)
            ρ_points = @.  cos_prof(z/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)

            ρ[:,j] = ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::TwoPhase2DLamCart, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)
    H = ub-lb

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:] .= x[i]
    end

    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*temperature/Tc)*H/L
        coef = sqrt(coef^2-1)/4
        for j in @chain(i)
            ρ_points = @.  cos_prof(X/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,nd+1,j) .= ρ_points
        end
    end
    return ρ
end


function initialize_profiles(model::EoSModel,structure::TwoPhase3DLamCart, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    H = ub-lb

    ngrid = structure.ngrid
    nd = dimension(structure)
    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end

    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*temperature/Tc)*H/L
        coef = sqrt(coef^2-1)/4

        for j in @chain(i)
            ρ_points = @.  cos_prof(X/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)

            selectdim(ρ,nd+1,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::TwoPhase2DHexCart, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    (pressure, temperature) = structure.conditions
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

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*temperature/Tc)/L
        for j in @chain(i)
            ρ_points = @. tanh_prof(r,ρ1[i],ρ2[i],R,coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,nd+1,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::TwoPhase3DHexCart, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end

    y = uniform_range(structure,2)
    Y = zeros(ngrid)

    for i in 1:ngrid[2]
        Y[:,i,:] .= y[i]
    end

    r = sqrt.(X.^2 + Y.^2)

    L = length_scale(model)
    H = ub-lb
    R = H/sqrt(2π)

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*temperature/Tc)/L
        for j in @chain(i)
            ρ_points = @. tanh_prof(r,ρ1[i],ρ2[i],R,coef)
            selectdim(ρ,nd+1,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::TwoPhase3DSphrCart, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = dimension(structure)
    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.ρbulk2

    pure = Clapeyron.split_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end

    y = uniform_range(structure,2)
    Y = zeros(ngrid)

    for i in 1:ngrid[2]
        Y[:,i,:] .= y[i]
    end

    z = uniform_range(structure,3)
    Z = zeros(ngrid)

    for i in 1:ngrid[3]
        Z[:,:,i] .= z[i]
    end

    r = sqrt.(X.^2 + Y.^2 + Z.^2)
  
    L = length_scale(model)
    H = ub-lb

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        (Tc, pc, vc) = crit_pure(pure[i])
        coef = (2.4728-2.3625*temperature/Tc)*H/L
        coef = sqrt(coef^2-1)/4
        for j in @chain(i)
            ρ_points = @. cos_prof(r / H, ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)

            selectdim(ρ,nd+1,j) .= ρ_points
        end
    end
    return ρ
end