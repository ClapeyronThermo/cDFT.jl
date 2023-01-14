
export initial_surface_tension_density_profile, initial_uniform_density_profile

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

function initial_interfacial_tension_density_profile(model::EoSModel,p,T,n,bounds,ngrid::Int64=101;coef=0.6*ones(length(model)),shift=zeros(length(model)))
    z = range(first(bounds),last(bounds),ngrid) |> collect

    (x,N,G) = tp_flash(model,p,T,n,RRTPFlash(K0=[100.,0.001]))

    v1 = volume(model,p,T,x[1,:])
    v2 = volume(model,p,T,x[2,:])

    ρ1 = x[1,:]./v1
    ρ2 = x[2,:]./v2

    ρ = DensityProfile[]
    for i in @comps
        boundary_conditions = [ρ2[i],ρ1[i]]

        σ = model.params.sigma[i]
        ρ_points =@. 1/2*(ρ1[i]-ρ2[i])*tanh(z/σ*coef[i]+shift[i])+1/2*(ρ1[i]+ρ2[i])

        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

function initial_uniform_density_profile(model::EoSModel,ρ_val,bounds,ngrid::Int64=101)
    z = range(first(bounds),last(bounds),ngrid) |> collect

    ρ = DensityProfile[]

    for i in @comps
        boundary_conditions = [ρ_val[i],ρ_val[i]]

        ρ_points = ρ_val[i]*ones(length(z))

        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

