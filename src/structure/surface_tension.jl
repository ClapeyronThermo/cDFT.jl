function initialize_profiles(model::EoSModel,structure::SurfaceTension1DCart)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, x) = structure.conditions

    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    Tc,vl,vv,x,y = _initial_eq_surface_tension(model,T,x)

    ρl = x./vl
    ρv = y./vv

    ρ = DensityProfile[]
    for i in @comps
        boundary_conditions = [ρv[i],ρl[i]]
        ρ_points =@. 1/2*(ρl[i]-ρv[i])*tanh(z/L*(2.4728-2.3625*T/Tc[i]))+1/2*(ρl[i]+ρv[i])

        push!(ρ,DensityProfile(ρ_points,z,[bounds[1],bounds[2]],boundary_conditions))
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