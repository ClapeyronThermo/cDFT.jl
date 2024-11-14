function initialize_profiles(model::EoSModel,structure::SurfaceTension1DCart, species)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, x) = structure.conditions

    z = uniform_range(structure) |> collect
    L = length_scale(model)
    lb,ub = first(z),last(z)
    midpoint = 0.5*(lb + ub)
    Tc,vl,vv,x,y = _initial_eq_surface_tension(model,T,x)

    ρl = x./vl
    ρv = y./vv
    shift_lo = (ub/4+3*lb/4)
    shift_hi = (3*ub/4+lb/4)
    ρ = zeros(ngrid,sum(species.nbeads))
    for i in @comps
        for j in @chain(i)
            coef_j = (2.4728-2.3625*T/Tc[i])/L
            ρ_points = @. ifelse(z<=midpoint,tanh_prof(z,ρl[i],ρv[i],shift_lo,coef_j),tanh_prof(z,ρl[i],ρv[i],shift_hi,coef_j))

            ρ[:,j] = ρ_points
        end
    end
    return ρ
end

function _initial_eq_surface_tension(model::EoSModel,T,x::SingleComp)
    (Tc,pc,vc) = crit_pure(model)
    (p,vl,vv) = saturation_pressure(model,T)
    return Tc,vl,vv,x,x
end

function _initial_eq_surface_tension(model::EoSModel,T,x)
    pure = Clapeyron.split_model(model)
    crit = crit_pure.(pure)
    Tc = first.(crit)
    (p,vl,vv,y) = bubble_pressure(model,T,x)
    return Tc,vl,vv,x,y 
end