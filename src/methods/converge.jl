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
    device = system.options.device
    
    δfδρ_res, model_cache, external_field_cache, propagator_cache = cDFT.preallocate(system, ρ)

    function obj(system,ln_G,ln_x)
        ln_x = reshape(ln_x, (ngrid..., nbeads))
        ln_Gx = reshape(ln_G, (ngrid..., nbeads))

        ρ .= exp.(ln_x)
         

        δFδρ_res!(system, ρ, δfδρ_res, model_cache...)
        
        evaluate_external_field!(system, ρ, δfδρ_res, external_field_cache)
        
        propagate!(system, ρ, δfδρ_res, propagator_cache)

        for i in @comps
            chem_pot_res_dens_i = log(species.bulk_density[i]) .+ 
                                            species.chempot_res[i]
            for k in @chain(i)
                if system.species.nbeads[i] != 1
                    α = findall(model.groups.n_intergroups[i][k,:] .== 1 .&& species.levels .> species.levels[k])
                else
                    α = k
                end

                # All operations stay vectorized on GPU
                selectdim(ln_Gx, nd+1, k) .=  chem_pot_res_dens_i .- 
                                            selectdim(δfδρ_res, nd+1, k)
            end
        end

        if any(typeof.(system.external_field) .<: ElectrostaticPotentialModel)
            ep_model = filter(x -> x isa ElectrostaticPotentialModel, system.external_field)[1]
            Z = model.charge

            psi_c = find_ψ_const(system.structure, ep_model, system.model, exp.(ln_Gx))/k_B/system.structure.conditions[2]
            for i in @comps
                for k in @chain(i)
                    selectdim(ln_Gx,nd+1,k) .-= psi_c*Z[k]
                end
            end
        end

        clamp!(ln_Gx, -100, Inf)

        ln_G = vec(ln_Gx)

        return ln_G
    end

    ln_GX0 = copy(ρ)
    f(ln_x) = obj(system, ln_GX0, ln_x)
    f!(ln_G, ln_x) = obj(system, ln_G, ln_x)

    ln_X0 = vec(log.(ρ))

    ρ_new = aasol(f!, ln_X0, 0, similar(ln_X0, length(ln_X0), 4); beta=1e-3, rtol=1e-4, atol=1e-4, maxit=10000, picard_maxit=1000, picard_beta = 1e-3, picard_rtol=1e-1, picard_atol=1e-1)

    ρ .= reshape(exp.(ρ_new.solution), (ngrid..., nbeads))
end

export converge!