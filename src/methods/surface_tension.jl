function surface_tension(model::EoSModel,T,x = SA[1.0];ngrid = 101)
    L = length_scale(model)
    (p,vl,vv,y) = bubble_pressure(model,T,x) #on single component, returns just the saturation pressure.
    # if T<0.75Tc
        ρ,z = initial_surface_tension_density_profile(model,T,x,[-15L,15L],201)
    # else
    #     ρ,z = initial_interfacial_density_profile(model,T,[-20σ,20σ],201)
    # end

    converge_profile!(model,ρ,T,z)
    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,vl,T,x)
    γ = F*k_B*T
    ∑x = sum(x)
    for i in @comps
        dz = ρ[i].mesh_size
        γ -= μ[i]*∫(ρ[i].density,dz) + p*∫(one.(z),dz)*x[i]/∑x
    end
    return γ
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

function initial_surface_tension_density_profile(model::EoSModel,T,x,bounds,ngrid::Int64=100)
    z = range(first(bounds),last(bounds),ngrid) |> collect
    L = length_scale(model)
    
    Tc,vl,vv,x,y = _initial_eq_surface_tension(model,T,x)

    ρl = x./vl
    ρv = y./vv

    ρ = DensityProfile[]
    for i in @comps
        boundary_conditions = [ρv[i],ρl[i]]
        ρ_points =@. 1/2*(ρl[i]-ρv[i])*tanh(z/L*(2.4728-2.3625*T/Tc[i]))+1/2*(ρl[i]+ρv[i])

        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

export surface_tension