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
    nbeads = length(system.profiles)
    species = system.species
    model = system.model
    method = system.options.solver
    ρ = system.profiles

    function obj(system,ln_G,ln_x)
        ln_x = reshape(ln_x, (ngrid, length(ρ)))
        ln_Gx = reshape(ln_G, (ngrid, length(ρ)))
        for i in 1:nbeads
            update_profile!(system.profiles[i], exp.(@view(ln_x[:,i])))
        end

        δfδρ_res = δFδρ_res(system)
        I1, I2 = 1.,1.

        species_id = 1
        bead_id = 1
        bead_1 = 1
        for i in 1:nbeads
            bead_n = species[species_id].nbeads
            if bead_id == 1
                I1, I2 = propagate(system, system.propagator, δfδρ_res[:,bead_1:bead_1+bead_n-1], species_id)
            end
            
            Threads.@threads for j in 1:ngrid
                ln_Gx[j,i] = log(species[species_id].bulk_density).+(species[species_id].chempot_res .- δfδρ_res[j,i]).+log(I1[j,bead_id].*I2[j,bead_id])
            end

            if bead_id == species[species_id].nbeads
                species_id += 1
                bead_1 += bead_n
                bead_id = 1
            else
                bead_id += 1
            end
        end
        
        ln_G = vec(ln_Gx)
        
        return ln_G
    end

    ln_X0 = zeros(ngrid,nbeads)
    ln_GX0 = copy(ln_X0)
    f(ln_x) = obj(system, ln_GX0, ln_x)
    
    for i in 1:nbeads
        ln_X0[:,i] = log.(ρ[i].density)
    end

    ln_X0 = vec(ln_X0)

    ρ_new = Solvers.fixpoint(f,ln_X0, method; rtol = 1e-4, max_iters = 100000)

    ρ_new = exp.(ρ_new)
    ρ_new = reshape(ρ_new,(ngrid,length(ρ)))
    for i in 1:nbeads
        update_profile!(system.profiles[i],ρ_new[:,i])
    end
end

export converge!