function surface_tension(model::EoSModel,T,x = [1.0])
    L = length_scale(model)
    (p,vl,vv,y) = bubble_pressure(model,T,x) #on single component, returns just the saturation pressure.
    # if T<0.75Tc
        ρ,z = initial_surface_tension_density_profile(model,T,x,[-10L,10L],101)
    # else
    #     ρ,z = initial_interfacial_density_profile(model,T,[-20σ,20σ],201)
    # end

    converge_profile!(model,ρ,T,z)
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
    return _initial_surface_tension_density_profile(
        model,:Cartesian,T,x,bounds,ngrid
    )
end

function initial_surface_tension_spherical_density_profile(model::EoSModel,T,x,bounds,ngrid::Int64=101)
    return _initial_surface_tension_density_profile(
        model,:Spherical,T,x,bounds,ngrid
    )
end

function _initial_surface_tension_density_profile(model::EoSModel,prof_type::Symbol,T,x,bounds,ngrid::Int64=101)
    DP = prof_type == :Cartesian ? CartesianDensityProfile : SphericalDensityProfile
    
    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)
    
    Tc,vl,vv,x,y = _initial_eq_surface_tension(model,T,x)

    ρl = x./vl
    ρv = y./vv

    ρ = DP[]
    for i in @comps
        boundary_conditions = [ρv[i],ρl[i]]
        ρ_points =@. 1/2*(ρl[i]-ρv[i])*tanh(z/L*(2.4728-2.3625*T/Tc[i]))+1/2*(ρl[i]+ρv[i])

        push!(ρ,DP(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

export surface_tension