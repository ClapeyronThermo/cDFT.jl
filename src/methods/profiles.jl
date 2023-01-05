function converge_profile!(model,ρ,T,z;method=NLSolvers.Anderson(0,50,0.02,nothing))
    ρl = ρ.boundary_conditions[2]
    μ_res = Clapeyron.VT_chemical_potential_res(model,1/ρl,T,[1.])/R̄/T

    # function obj(model,ρ,T,z,Gx,x,μ_res,ρl,α)
    #     ρ = ClapeyronDFT.update_profile!(ρ,x)
    #     Gx .= (1-α).*x+α.*ρl.*exp.(μ_res.-δFδρ_res(model,ρ,T,z))
    #     println(NLSolvers.norm(Gx.-x))
    #     return Gx
    # end

    function obj(model,ρ,T,z,x,μ_res,ρl,α)
        ρ = ClapeyronDFT.update_profile!(ρ,x)
        Gx = (1-α).*x+α.*ρl.*exp.(μ_res.-δFδρ_res(model,ρ,T,z))
        return Gx
    end

    fX(x) = obj(model,ρ,T,z,x,μ_res,ρl,0.05)

    r = fixed_point(fX, deepcopy(ρ.density);Algorithm = :Anderson, 
                                            ConvergenceMetric = norm(output,input) = maximum(abs.(output./input .-1)),
                                            ConvergenceMetricThreshold=1e-6,
                                            MaxM=50)
    return update_profile!(ρ,r.FixedPoint_)
end