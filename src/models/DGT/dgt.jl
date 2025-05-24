struct DGTSpecies <: DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

function f_res(system::DGTSystem, model::EoSModel,n)
    nd = dimension(system)
    n1,n2 = @view(n[1,:]),@view(n[2:2+nd-1,:])
    (p,T) = system.structure.conditions
    ∑ρ = sum(n1)
    return a_res(model, 1., T, n1./N_A)*∑ρ + ∇a_res(system,system.gradient,n1,n2)
end

function ∇a_res(system::DGTSystem,gradient::GradientModel, ρ̄, ∇ρ̄)
    (p,T) = system.structure.conditions
    κ = kappa(gradient, T, ρ̄)
    _res = zero(eltype(ρ̄))
    for i in 1:length(ρ̄)
        _res += κ[i,i]*dot(∇ρ̄[:,i],∇ρ̄[:,i])/2
        for j in i+1:length(ρ̄)
            _res += κ[i,j]*dot(∇ρ̄[:,i],∇ρ̄[:,j])
        end
    end
    return _res/T
end

include("gradients/const.jl")