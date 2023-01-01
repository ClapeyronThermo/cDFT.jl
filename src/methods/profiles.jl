function converge_profile!(model,ρ,T,z;method=NLSolvers.Anderson(50,1, nothing,nothing))
    ρl = ρ.boundary_conditions[2]
    μ_res = Clapeyron.VT_chemical_potential_res(model,1/ρl,T,[1.])/R̄/T

    function obj(model,ρ,T,z,Gx,x,μ_res,ρl,α)
        ρ = ClapeyronDFT.update_profile!(ρ,x)
        Gx .= (1-α).*x+α.*ρl.*exp.(μ_res.-δFδρ_res(model,ρ,T,z))
    end

    fX = (Gx,x) -> obj(model,ρ,T,z,Gx,x,μ_res,ρl,0.02)

    r = NLSolvers.fixedpoint!(fX, deepcopy(ρ.density), method; maxiter=1000)
    return update_profile!(ρ,r.x)
end