function converge_profile!(model,ρ,T,z,method=AndersonFixPoint(picard_damping=1e-3,damping=5e-2,memory=50,delay=100,drop_tol=Inf))
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
        # center_profile!(ρ)

        δfδρ_res = δFδρ_res(model,ρ,T,z)
        for i in @comps
            ln_xi = @view(ln_x[:,i])
            δfδρ_resi = @view(δfδρ_res[:,i])
            ln_Gx[:,i] .= (1 .- α) .* ln_xi .+ α .* (ln_ρl[i].+(μ_res[i] .- δfδρ_resi))
        end

        ln_G = vec(ln_Gx)
        
        return ln_G
    end

    ln_X0 = zeros(length(z),length(ρ))
    ln_GX0 = copy(ln_X0)
    f(ln_x) = obj(model, ln_GX0, ρ, T, z, ln_x, μ_res, ρl, 1.)
    
    for i in @comps
        ln_X0[:,i] = log.(ρ[i].density)
    end
    ln_X0 = vec(ln_X0)

    ρ_new = Solvers.fixpoint(f,ln_X0,method,rtol = 1e-4, max_iters = 10000)

    ρ_new = exp.(ρ_new)
    ρ_new = reshape(ρ_new,(length(z),length(ρ)))
    for i in @comps
        ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
    end

    # r = fixed_point(fX, ln_X0;Algorithm = :Anderson, 
    #                             ConvergenceMetric = norm(output,input) = maximum(abs.(output./input .-1)/damping),
    #                             ConvergenceMetricThreshold=1e-5,
    #                             MaxIter=10000,
    #                             MaxM=25)
    
    # if ismissing(r.FixedPoint_)
    #     throw(error("Convergence failed"))
    #     ρ_new = exp.(r.Outputs_[:,end])
    #     ρ_new = reshape(ρ_new,(length(z),length(ρ)))
    #     for i in @comps
    #         ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
    #     end
    # else
    #     ρ_new = exp.(r.FixedPoint_)
    #     ρ_new = reshape(ρ_new,(length(z),length(ρ)))
    #     for i in @comps
    #         ρ[i] = update_profile!(ρ[i],ρ_new[:,i])
    #     end
    # end
end

function center_profile!(ρ)
    shift_idx = 0
    z = ρ[1].coords
    idx_interface = findfirst(z.>=0.)
    for j in 1:length(ρ)
        boundary_conditions = ρ[j].boundary_conditions
        ρ_interface = (boundary_conditions[1]+boundary_conditions[2])/2
        if ρ_interface>boundary_conditions[1]
            shift_idx += findfirst(ρ[j].density.>ρ_interface)
        else
            shift_idx += findfirst(ρ[j].density.<ρ_interface)
        end
    end
    shift_idx = Int(round(shift_idx/length(ρ)))

    # println(idx_interface)
    # println(shift_idx)
    didx = idx_interface - shift_idx
    # println(didx)

    if didx != 0
        for i in 1:length(ρ)
            boundary_conditions = ρ[i].boundary_conditions
            ρold = ρ[i].density
            ρnew = zeros(length(ρold))
            if didx < 0
                ρnew[1:end+didx] = ρold[-didx+1:end]
                ρnew[end+didx+1:end] .= boundary_conditions[2]
            else
                ρnew[didx:end] = ρold[1:end-didx+1]
                ρnew[1:didx-1] .= boundary_conditions[1]
            end
            update_profile!(ρ[i],ρnew)
        end
    end
end

