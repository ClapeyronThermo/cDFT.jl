function converge!(system::DFTSystem)
    (p, T, z) = system.structure.conditions
    ngrid = system.structure.ngrid
    model = system.model
    method = system.options.solver
    ρ = system.profiles

    ρl =[ρ[i].boundary_conditions[2] for i in @comps]
    Vl = 1/sum(ρl)
    X = ρl./sum(ρl)
    μ_res = Clapeyron.VT_chemical_potential_res(model,Vl,T,X)/R̄/T

    function obj(system,ln_G,ln_x)
        ln_x = reshape(ln_x, (ngrid, length(ρ)))
        ln_Gx = reshape(ln_G, (ngrid, length(ρ)))
        ln_ρl = log.(ρl)
        for i in @comps
            update_profile!(system.profiles[i], exp.(@view(ln_x[:,i])))
        end

        δfδρ_res = δFδρ_res(system)
        for i in @comps
            Threads.@threads for j in 1:ngrid
                ln_Gx[j,i] = ln_ρl[i].+(μ_res[i] .- δfδρ_res[j,i])
            end
        end

        ln_G = vec(ln_Gx)
        
        return ln_G
    end

    ln_X0 = zeros(ngrid,length(ρ))
    ln_GX0 = copy(ln_X0)
    f(ln_x) = obj(system, ln_GX0, ln_x)
    
    for i in @comps
        ln_X0[:,i] = log.(ρ[i].density)
    end

    ln_X0 = vec(ln_X0)

    ρ_new = Solvers.fixpoint(f,ln_X0, method; rtol = 1e-4, max_iters = 10000)

    ρ_new = exp.(ρ_new)
    ρ_new = reshape(ρ_new,(ngrid,length(ρ)))
    for i in @comps
        update_profile!(system.profiles[i],ρ_new[:,i])
    end
end

export converge!