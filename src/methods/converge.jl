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

        ln_x = Adapt.adapt(device, ln_x)
        ln_Gx = Adapt.adapt(device, ln_Gx)
        
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

        # if hasfield(typeof(system), :external_field)
        #     psi_c = find_ψ_const(system.structure, system.external_field, system.model, exp.(ln_Gx), Z)/k_B/system.structure.conditions[2]
        #     for i in @comps
        #         for k in @chain(i)
        #             selectdim(ln_Gx,nd+1,k) .-= psi_c*model.charge[k]
        #         end
        #     end
        # end

        clamp!(ln_Gx, -100, Inf)
        
        ln_G = vec(ln_Gx)
        # println(ln_G)
        # If any < -100, set to -100 to avoid overflow

        ln_G = Adapt.adapt(CPU(), ln_G)
        
        return ln_G
    end

    ln_GX0 = copy(ρ)
    f(ln_x) = obj(system, ln_GX0, ln_x)
    f!(ln_G, ln_x) = obj(system, ln_G, ln_x)

    ln_X0 = Adapt.adapt(CPU(), vec(log.(ρ)))

    ρ_new = SIAMFANLEquations.aasol(f!, ln_X0, 0, zeros(length(ln_X0),4); beta=1e-3, rtol=1e-1, atol=1e-1, maxit=1000)
    ρ_new = SIAMFANLEquations.aasol(f!, ρ_new.solution, 5, zeros(length(ln_X0),14); beta=1e-3, rtol=1e-4, atol=1e-4, maxit=10000)
    # println(ρ_new.history)

    ρ .= Adapt.adapt(device, reshape(exp.(ρ_new.solution),(ngrid...,nbeads)))
end

export converge!