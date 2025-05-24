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
function converge!(system::Union{DFTSystem,DGTSystem},ρ)
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nbeads = size(ρ,nd+1)
    species = system.species
    model = system.model
    method = system.options.solver
    Z = get_coords(system.structure)

    function obj(system,ln_G,ln_x)
        ln_x = reshape(ln_x, (ngrid..., nbeads))
        ln_Gx = reshape(ln_G, (ngrid..., nbeads))
        
        ρ .= exp.(ln_x)

        δfδρ_res = δFδρ_res(system, ρ)
        Gcα, Gp = propagate(system, δfδρ_res, ρ)
        Vext = evaluate_external_field(system, ρ, Z)
        # println(δfδρ_res)
        # println(species.chempot_res)
        for i in @comps
            for k in @chain(i)
                if system.species.nbeads[i] != 1
                    α = findall(model.groups.n_intergroups[i][k,:] .== 1 .&& species.levels .> species.levels[k])
                else
                    α = k
                end

                for j in Iterators.product([1:ngrid[i] for i in 1:length(ngrid)]...)
                    ln_Gx[j...,k] = log(species.bulk_density[i]) + (species.chempot_res[i] - δfδρ_res[j...,k]) + log(Gp[j...,k]) + sum(log.(Gcα[j...,k,α])) - Vext[j...,k]
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

    ρ .= reshape(exp.(ρ_new),(ngrid...,nbeads))
end

export converge!