function interfacial_tension(model::EoSModel,p,T,n)
    L = length_scale(model)

    ρ,z = initial_interfacial_tension_density_profile(model,p,T,n,[-10L,10L],201)

    converge_profile!(model,ρ,T,z,AndersonFixPoint(delay=100,memory=50,damping=1e-2,picard_damping=1e-3,drop_tol=Inf))

    return eval_interfacial_tension(model,p,T,ρ,z)
end

function initial_interfacial_tension_density_profile(model::EoSModel,p,T,n,bounds,ngrid::Int64=101)
    nc = length(model)    

    z = range(first(bounds),last(bounds),ngrid) |> collect

    (x,N,G) = tp_flash(model,p,T,n,RRTPFlash(equilibrium=:lle))

    v1 = volume(model,p,T,x[1,:])
    v2 = volume(model,p,T,x[2,:])

    ρ1 = x[1,:]./v1
    ρ2 = x[2,:]./v2

    f(x) = obj_initial_profile(model,p,T,ρ1,ρ2,bounds,z,x[1:nc],x[nc+1:2*nc])
    k0 = ones(nc)
    c0 = zeros(nc)
    z0 = vcat(k0,c0)
    #res =  optimize(f,z0,Optim.NelderMead(),Optim.Options())
    #x_sol = Optim.minimizer(res)
    scalarobj = NLSolvers.ScalarObjective(f = f)
    optprob = NLSolvers.OptimizationProblem(obj = scalarobj,inplace = false,bounds = nothing)
    res = NLSolvers.solve(optprob,z0,NLSolvers.NelderMead(),NLSolvers.OptimizationOptions())
    x_sol = Solvers.x_sol(res)
    k = x_sol[1:nc]
    c = x_sol[nc+1:2*nc]
    ρ,z = _initial_interfacial_tension_density_profile(model,ρ1,ρ2,bounds,z,k,c)        
    
    return ρ, z
end


# ρ(z) = 1/2(ρ1-ρ2)*tanh(a*z/σ+b)+1/2*(ρ1+ρ2)+c^2*exp(-(z/\sigma))
function _initial_interfacial_tension_density_profile(model::EoSModel,ρ1,ρ2,bounds,z,coef=ones(length(model)),shift=zeros(length(model)))
    L = length_scale(model)

    ρ = DensityProfile[]
    for i in @comps
        boundary_conditions = [ρ2[i],ρ1[i]]

        ρ_points =@. 1/2*(ρ1[i]-ρ2[i])*tanh(z/L*coef[i]+shift[i])+1/2*(ρ1[i]+ρ2[i])

        push!(ρ,DensityProfile(ρ_points,z,bounds,boundary_conditions))
    end
    return ρ, z
end

function obj_initial_profile(model,p,T,ρ1,ρ2,bounds,z,k,c)
    ρ, z = _initial_interfacial_tension_density_profile(model,ρ1,ρ2,bounds,z,k,c)
    return eval_interfacial_tension(model,p,T,ρ,z)
end

export interfacial_tension