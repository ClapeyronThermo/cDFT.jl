function converge_profile!(model,ρ,T,z;damping=0.05)
    ρl =[ρ[i].boundary_conditions[2] for i in @comps]
    Vl = 1/sum(ρl)
    X = ρl./sum(ρl)
    μ_res = Clapeyron.VT_chemical_potential_res(model,Vl,T,X)/R̄/T

    function obj(model,ln_G,ρ,T,z,ln_x,μ_res,ρl,α)
        ln_x = reshape(ln_x, (length(z), length(ρ)))
        ln_Gx = reshape(ln_G, (length(z), length(ρ)))
        ln_ρl = log.(ρl)
        for i in @comps
            ρ[i] = update_profile!(ρ[i], exp.(@view(ln_x[:,i])))
        end

        δfδρ_res = δFδρ_res(model,ρ,T,z)
        for i in @comps
            ln_xi = @view(ln_x[:,i])
            δfδρ_resi = @view(δfδρ_res[:,i])
            ln_Gx[:,i] .= (1 .- α) .* ln_xi .+ α .* (ln_ρl[i].+(μ_res[i] .- δfδρ_resi))
        end
        
        return ln_G
    end

    ln_X0 = zeros(length(z),length(ρ))
    ln_GX0 = copy(ln_X0)
    fX(ln_x) = obj(model, copy(ln_x), ρ, T, z, ln_x, μ_res, ρl, damping)
    
    for i in @comps
        ln_X0[:,i] = log.(ρ[i].density)
    end

    ln_X0 = vec(ln_X0)

    # ρ_new = Solvers.fixpoint(f!,X0,AndersonFixPoint(memory =50),rtol = 1e-4)

    r = fixed_point(fX, ln_X0;Algorithm = :Anderson, 
                                            ConvergenceMetric = norm(output,input) = maximum(abs.(output./input .-1)),
                                            ConvergenceMetricThreshold=1e-5,
                                            MaxM=50)
    
    if isempty(r.FixedPoint_)
        warning("Convergence failed")
        ρ_new = exp.(r.Outputs_[:,end])
        ρ_new = reshape(ρ_new,(length(z),length(ρ)))
        for i in @comps
            ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
        end
    else
        ρ_new = exp.(r.FixedPoint_)
        ρ_new = reshape(ρ_new,(length(z),length(ρ)))
        for i in @comps
            ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
        end
    end
end