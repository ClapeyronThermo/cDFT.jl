function interfacial_tension(model::EoSModel,p,T,n)
    σ = maximum(model.params.sigma.values)

    ρ,z = initial_interfacial_tension_density_profile(model,p,T,n,[-10σ,10σ],201;coef=[2.5,3.5])

    ρ = converge_profile!(model,ρ,T,z;damping=0.001)

    ρ_bound = [ρ[i].boundary_conditions[1] for i in @comps]

    v = 1/sum(ρ_bound)
    x = ρ_bound/sum(ρ_bound)

    F = F_tot(model,ρ,T,z)
    μ = Clapeyron.VT_chemical_potential(model,v,T,x)
    return F*k_B*T-sum([μ[i]*∫(ρ[i].density,ρ[i].mesh_size) for i in @comps])+p*∫(one.(z),ρ[1].mesh_size)
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

export interfacial_tension