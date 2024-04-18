function F_res(system::DFTSystem)
    model = system.model
    dz = system.profiles[1].mesh_size
    ngrid = system.structure.ngrid

    n = evaluate_field(system)

    f(x) = f_res(system,model,x)

    ϕ = zeros(ngrid)

    Threads.@threads for i in 1:ngrid
        ϕ[i] = f(@view n[i,:,:])
    end

    return ∫(ϕ,dz)
end

function δFδρ_res(system::DFTSystem)
    model = system.model
    fields = system.fields
    ρ = system.profiles
    z = ρ[1].coords
    nc = length(model)
    nf = length(fields)
    dz = system.profiles[1].mesh_size
    ngrid = system.structure.ngrid

    n = evaluate_field(system)

    f(x) = f_res(system,model,x)
    df(x) = ForwardDiff.gradient(f,x)

    δf = zeros(ngrid,nf,nc)

    # dx = similar(n,nf)
    Threads.@threads for i in 1:ngrid
        δf[i,:,:] = df(n[i,:,:])
    end

    δFδρ_res = zeros(ngrid,nc)

    for j in 1:nf
        ∂f = DensityProfile[]
        for i in @comps
            lim = system.species.size[i]
            bounds = ρ[i].bounds.+(-lim,lim)
            boundary_conditions = ρ[i].boundary_conditions
            bc1 = typeof(boundary_conditions[1])(δf[1,j,i],-1)
            bc2 = typeof(boundary_conditions[1])(δf[end,j,i],1)
            push!(∂f, DensityProfile(@view(δf[:,j,i]),z,bounds,(bc1,bc2)))
        end
        δFδρ_res += integrate_field(system,fields[j],∂f)
    end

    return δFδρ_res
end

include("assoc.jl")