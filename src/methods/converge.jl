"""
    converge!(system::DFTSystem)

For a given system, converge the profiles using the solver specified under `system.options.solver`. Convergence is achieved by solving the generic equation:
```julia
ρi = ρi_bulk*exp(β(μi_res - δFδρ_res))
```
For stability purposes, the equation has be reformulated as:
```julia
ln(ρi) = ln(ρi_bulk) + β(μi_res - δFδρ_res)
```
The default solver uses Anderson Mixing with 100 initial Picard iterations, 50 memory points, 1e-2 damping, and an infinite drop tolerance. 
"""
function converge!(system::DFTSystem)
    (p, T, z) = system.structure.conditions
    ngrid = system.structure.ngrid
    nbeads = system.species.nbeads
    model = system.model
    method = system.options.solver
    ρ = system.profiles

    ρl =[ρ[i].boundary_conditions[2].value for i in @comps]
    Vl = 1/sum(ρl)
    X = ρl./sum(ρl)
    μ_res = Clapeyron.VT_chemical_potential_res(model,Vl,T,X)/R̄/T

    function obj(system,ln_G,ln_x)
        ln_x = reshape(ln_x, (ngrid, length(ρ)))
        ln_Gx = reshape(ln_G, (ngrid, length(ρ)))
        ln_ρl = log.(ρl)
        for i in 1:sum(nbeads)
            update_profile!(system.profiles[i], exp.(@view(ln_x[:,i])))
        end

        δfδρ_res = δFδρ_res(system)
        for i in @comps
            I1, I2 = propagate(system, δfδρ_res, i)
            bead_id = findall(system.species.species_id .== i)
            for k in 1:nbeads[i]
                id = bead_id[k]
                Threads.@threads for j in 1:ngrid
                    ln_Gx[j,id] = ln_ρl[i].+(μ_res[i] .- δfδρ_res[j,id]).+log(I1[j,k].*I2[j,k])
                end
            end
        end
        
        ln_G = vec(ln_Gx)
        
        return ln_G
    end

    ln_X0 = zeros(ngrid,length(ρ))
    ln_GX0 = copy(ln_X0)
    f(ln_x) = obj(system, ln_GX0, ln_x)
    
    for i in 1:sum(nbeads)
        ln_X0[:,i] = log.(ρ[i].density)
    end

    ln_X0 = vec(ln_X0)

    ρ_new = Solvers.fixpoint(f,ln_X0, method; rtol = 1e-4, max_iters = 100000)

    ρ_new = exp.(ρ_new)
    ρ_new = reshape(ρ_new,(ngrid,length(ρ)))
    for i in @comps
        update_profile!(system.profiles[i],ρ_new[:,i])
    end
end

export converge!