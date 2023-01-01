function converge_profile!(model,ρ,T,z)
    ρl = ρ.boundary_conditions[2]
    μ_res = Clapeyron.VT_chemical_potential_res(model,1/ρl,T,[1.])/R̄/T

    function fX(out,in)
        ρ = update_profile!(ρ,in)
        out .= ρl.*exp.(μ_res.-δFδρ_res(model,ρ,T,z))
        return out
    end

    ρ_conv = Solvers.fixpoint(fX,ρ.density,Solvers.SSFixPoint(0.05),atol=1e-6,rtol=1e-6,max_iters=1000)
    return update_profile!(ρ,ρ_conv)
end