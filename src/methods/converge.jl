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
function converge!(system::AbstractcDFTSystem,ρ)
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
        Vext = evaluate_external_field(system, ρ, Z)
        δfδρ_res .+= Vext
        Gcα, Gp = propagate(system, δfδρ_res, ρ)
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
                    ln_Gx[j...,k] = log(species.bulk_density[i]) + (species.chempot_res[i] - δfδρ_res[j...,k]) + log(Gp[j...,k]) + sum(log.(Gcα[j...,k,α]))
                end
            end
        end

        if hasfield(typeof(system), :external_field)
            psi_c = find_ψ_const(system.structure, system.external_field, system.model, exp.(ln_Gx), Z)/k_B/system.structure.conditions[2]
            for i in @comps
                for k in @chain(i)
                    selectdim(ln_Gx,nd+1,k) .-= psi_c*model.charge[k]
                end
            end
        end

        
        
        ln_G = vec(ln_Gx)
        # If any < -100, set to -100 to avoid overflow
        ln_G[ln_G .< -100] .= -100
        
        return ln_G
    end

    ln_GX0 = copy(ρ)
    f(ln_x) = obj(system, ln_GX0, ln_x)
    f!(ln_G, ln_x) = obj(system, ln_G, ln_x)

    ln_X0 = vec(log.(ρ))

    ρ_new = SIAMFANLEquations.aasol(f!,ln_X0, 0, zeros(length(ln_X0),4); beta=1e-3, rtol=1e-1, atol=1e-1, maxit=1000)
    ρ_new = SIAMFANLEquations.aasol(f!,ρ_new.solution, 5, zeros(length(ln_X0),14); beta=1e-3, rtol=1e-4, atol=1e-4, maxit=10000)
    # println(ρ_new.history)

    ρ .= reshape(exp.(ρ_new.solution),(ngrid...,nbeads))
end

export converge!