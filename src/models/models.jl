function F_res(system::DFTSystem)
    model = system.model
    dz = system.profiles[1].mesh_size
    ngrid = system.structure.ngrid

    n = evaluate_field(system)

    f(x) = f_res(system,model,x)

    ϕ = zeros(ngrid)

    Threads.@threads for i in 1:ngrid
        ϕ[i] = f(@view n[:,:,i])
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

    δf = zeros(nf,nc,ngrid)

    # dx = similar(n,nf)
    Threads.@threads for i in 1:ngrid
        δf[:,:,i] = df(n[:,:,i])
    end

    δFδρ_res = zeros(nc,ngrid)

    for j in 1:nf
        ∂f = DensityProfile[]
        for i in @comps
            lim = system.species.size[i]
            bounds = ρ[i].bounds.+(-lim,lim)
            push!(∂f, DensityProfile(@view(δf[j,i,:]),z,bounds,[δf[j,i,1],δf[j,i,end]]))
        end
        δFδρ_res += integrate_field(system,fields[j],∂f)
    end

    return δFδρ_res
end


include("assoc.jl")

