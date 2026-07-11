function __coeff_cos_prof_correlation(pure,T,scale = one(T))
    Tc,_,_ = crit_pure(model)
    Tr = T/Tc
    c0 = 2.4728 - 2.3625*Tr
    return c0 * scale
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{1,Cartesian,TwoPhaseSystem{:Cartesian}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub-lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure) |> collect
    X = collect(x)

    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,H/L)
        coef = sqrt(coef^2-1)/4
        for j in @chain(i)
            ρ_points = @. cos_prof(X/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ[:,j] = ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{2,Cartesian,TwoPhaseSystem{:Lamellar}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:] .= x[i]
    end

    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,H/L)
        coef = sqrt(coef^2-1)/4
        for j in @chain(i)
            ρ_points = @.  cos_prof(X/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,3,j) .= ρ_points
        end
    end
    return ρ
end


function initialize_profiles(model::EoSModel,structure::DFTStructure{3,Cartesian,TwoPhaseSystem{:Lamellar}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

    x = uniform_range(structure,1)
    X = zeros(ngrid)

    for i in 1:ngrid[1]
        X[i,:,:] .= x[i]
    end

    L = length_scale(model)

    ρ = allocate(device, FP, ngrid..., sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,H/L)
        coef = sqrt(coef^2-1)/4

        for j in @chain(i)
            ρ_points = @.  cos_prof(X/(ub-lb), ρ1[i], ρ2[i], (ub / 4 + 3 * lb / 4), coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,4,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{2,Cartesian,TwoPhaseSystem{:HexLattice}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

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
    R = H/sqrt(2π)

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,1/L)
        for j in @chain(i)
            ρ_points = @. tanh_prof(r,ρ1[i],ρ2[i],R,coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,3,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{3,Cartesian,TwoPhaseSystem{:Lamellar}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

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
    R = H/sqrt(2π)

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,1/L)
        for j in @chain(i)
            ρ_points = @. tanh_prof(r,ρ1[i],ρ2[i],R,coef)
            selectdim(ρ,4,j) .= ρ_points
        end
    end
    return ρ
end

function initialize_profiles(model::EoSModel,structure::DFTStructure{3,Cartesian,TwoPhaseSystem{:Spherical}}, species, device, ::Type{FP}=Float64) where FP<:AbstractFloat
    lb,ub = bounds(structure,1)
    H = ub - lb
    mb = 0.5*(lb + ub)
    ngrid = structure.ngrid
    nd = length(ngrid)

    (pressure, temperature) = structure.conditions
    ρ1 = structure.ρbulk
    ρ2 = structure.topology.ρbulk2

    pure = Clapeyron.split_pure_model(model)

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
    R = H*(3/(8π))^(1/3)

    ρ = allocate(device, FP, ngrid...,sum(species.nbeads))
    for i in @comps
        coef = __coeff_cos_prof_correlation(pure[i],temperature,1/L)
        for j in @chain(i)
            ρ_points = @. tanh_prof(r, ρ1[i], ρ2[i], R, coef)
            ρ_points = adapt_to_device(device, FP, ρ_points)
            selectdim(ρ,4,j) .= ρ_points
        end
    end
    return ρ
end