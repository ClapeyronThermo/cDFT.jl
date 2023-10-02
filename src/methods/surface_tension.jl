function surface_tension(model::EoSModel,T,x = [1.0])
    σ = maximum(model.params.sigma.values)
    (p,vl,vv,y) = bubble_pressure(model,T,x) #on single component, returns just the saturation pressure.
    # if T<0.75Tc
        ρ,z = initial_surface_tension_density_profile(model,T,x,[-10σ,10σ],101)
    # else
    #     ρ,z = initial_interfacial_density_profile(model,T,[-20σ,20σ],201)
    # end

    ρ = converge_profile!(model,ρ,T,z)
    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(one.(z),ρ[1].mesh_size)
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

function initial_surface_tension_density_profile(model::EoSModel,T,x,bounds,ngrid::Int64=101)
    z = range(first(bounds),last(bounds),ngrid) |> collect
    
    Tc,vl,vv,x,y = _initial_eq_surface_tension(model,T,x)

    ρl = x./vl
    ρv = y./vv

    ρ = DensityProfile[]
    for i in @comps
        boundary_conditions = [ρv[i],ρl[i]]
        σ = model.params.sigma[i]
        ρ_points =@. 1/2*(ρl[i]-ρv[i])*tanh(z/σ*(2.4728-2.3625*T/Tc[i]))+1/2*(ρl[i]+ρv[i])

        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

export surface_tension