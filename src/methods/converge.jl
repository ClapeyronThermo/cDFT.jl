"""
    converge!(system::DFTSystem, ρ)

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
function converge!(system::DFTSystem,ρ)
    ngrid = system.structure.ngrid
    nbeads = size(ρ,2)
    species = system.species
    model = system.model
    method = system.options.solver
    z = vec(LinRange(system.structure.bounds[1],system.structure.bounds[2],ngrid))

    function obj(system,ln_G,ln_x)
        ln_x = reshape(ln_x, (ngrid, nbeads))
        ln_Gx = reshape(ln_G, (ngrid, nbeads))
        
        ρ .= exp.(ln_x)

        δfδρ_res = δFδρ_res(system, ρ)
        Gcα, Gp = propagate(system, system.propagator, δfδρ_res, ρ)
        Vext = evaluate_external_field(system, ρ, z)
        # println(δfδρ_res)
        # println(species.chempot_res)
        for i in @comps
            for k in @chain(i)
                if system.species.nbeads[i] != 1
                    α = findall(model.groups.n_intergroups[i][k,:] .== 1 .&& species.levels .> species.levels[k])
                else
                    α = k
                end

                Threads.@threads for j in 1:ngrid
                    ln_Gx[j,k] = log(species.bulk_density[i]) + (species.chempot_res[i] - δfδρ_res[j,k]) + log(Gp[j,k]) + sum(log.(Gcα[j,k,α])) - Vext[j,k]
                end
            end
        end
        ln_G = vec(ln_Gx)
        
        return ln_G
    end

    ln_GX0 = copy(ρ)
    f(ln_x) = obj(system, ln_GX0, ln_x)

    ln_X0 = vec(log.(ρ))

    ρ_new = Solvers.fixpoint(f,ln_X0, method; rtol = 1e-4, max_iters = 100000)

    ρ .= reshape(exp.(ρ_new),(ngrid,nbeads))
end

export converge!