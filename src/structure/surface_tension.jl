function initialize_profiles(model::EoSModel,structure::SurfaceTension1DCart, species)
    bounds = structure.bounds
    ngrid = structure.ngrid
    (p, T, x) = structure.conditions

    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)

    Tc,vl,vv,x,y = _initial_eq_surface_tension(model,T,x)

    ρl = x./vl
    ρv = y./vv

    ρ = zeros(ngrid,sum(species.nbeads))
    for i in @comps
        for j in @chain(i)
            ρ_points = @. tanh_prof(z,ρl[i],ρv[i],(bounds[2]/4+3*bounds[1]/4),(2.4728-2.3625*T/Tc[i])/L)*(z<=(bounds[2]+bounds[1])/2) +
                          tanh_prof(z,ρv[i],ρl[i],(3*bounds[2]/4+bounds[1]/4),(2.4728-2.3625*T/Tc[i])/L)*(z>(bounds[2]+bounds[1])/2)

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