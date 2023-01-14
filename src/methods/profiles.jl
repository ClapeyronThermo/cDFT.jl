function converge_profile!(model,ρ,T,z;damping=0.05)
    ρl =[ρ[i].boundary_conditions[2] for i in @comps]
    Vl = 1/sum(ρl)
    X = ρl./sum(ρl)
    μ_res = Clapeyron.VT_chemical_potential_res(model,Vl,T,X)/R̄/T

    # function obj(model,ρ,T,z,Gx,x,μ_res,ρl,α)
    #     ρ = ClapeyronDFT.update_profile!(ρ,x)
    #     Gx .= (1-α).*x+α.*ρl.*exp.(μ_res.-δFδρ_res(model,ρ,T,z))
    #     println(NLSolvers.norm(Gx.-x))
    #     return Gx
    # end

    function obj(model,G,ρ,T,z,x,μ_res,ρl,α)
        x = reshape(x,(length(z),length(ρ)))
        Gx = reshape(G,(length(z),length(ρ)))
        for i in @comps
            ρ[i] = update_profile!(ρ[i],@view(x[:,i]))
        end

        δfδρ_res = δFδρ_res(model,ρ,T,z)
        #Gx = zeros(length(z),length(ρ))
        for i in @comps
            xi = @view(x[:,i])
            δfδρ_resi = @view(δfδρ_res[:,i])
            Gx[:,i] .= (1 .- α) .* xi .+ α .* ρl[i].*exp.(μ_res[i] .- δfδρ_resi)
        end
        return G
    end

    X0 = zeros(length(z),length(ρ))
    GX0 = copy(X0)
    f!(Gx,x) = obj(model,Gx,ρ,T,z,x,μ_res,ρl,damping)
    fX(x) = obj(model,copy(x),ρ,T,z,x,μ_res,ρl,damping)
    
    for i in @comps
        X0[:,i] = ρ[i].density
    end

    X0 = vec(X0)

    ρ_new = Solvers.fixpoint(f!,X0,AndersonFixPoint(memory =50),rtol = 1e-4)
    #=r = fixed_point(fX, X0;Algorithm = :Anderson, 
                                            ConvergenceMetric = norm(output,input) = maximum(abs.(output./input .-1)),
                                            ConvergenceMetricThreshold=1e-4,
                                            MaxM=5)
    # return r =#
    #ρ_new = r.FixedPoint_
    ρ_new = reshape(ρ_new,(length(z),length(ρ)))
    for i in @comps
        ρ[i] = ClapeyronDFT.update_profile!(ρ[i],ρ_new[:,i])
    end
    return ρ
end