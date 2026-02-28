abstract type ElectrostaticPotentialModel <: ExternalFieldModel end

struct ElectrostaticPotential{M,P,iP} <: ElectrostaticPotentialModel
    ϵ_r::Float64
    map::M
    plan::P
    iplan::iP
end

export ElectrostaticPotential

function ElectrostaticPotential(model::ElectrolyteModel, structure::DFTStructure)
    (_, T) = structure.conditions
    ρbulk = structure.ρbulk
    ϵ_r = dielectric_constant(model.ionmodel.RSPmodel, 1., T, ρbulk)
    ngrid = structure.ngrid     
    nd = length(ngrid)
   
    ω = structure_ω(structure)

    map = 1 ./(4*pi^2*sum(ω.^2,dims=3))*N_A*e_c^2/ϵ_0/ϵ_r .+ 0.0im
    map[1] = 0.0
    plan = plan_fft(selectdim(map,nd+1,1), 1:length(ngrid))
    iplan = inv(plan)
    return ElectrostaticPotential(ϵ_r, map, plan, iplan)
end


function evaluate_external_field(structure::DFTStructure,external_field::ElectrostaticPotentialModel,model::ElectrolyteModel,ρ::Array{Float64},z)
    T = structure.conditions[2]
    Z = model.charge
    ngrid = structure.ngrid
    bounds = structure.bounds
    L = bounds[2] - bounds[1]
    Vol = prod(L)
    nbeads = length(Z)
    nd = length(ngrid)
    # obtain charge profiles
    q = zeros(ngrid...)
    for i in 1:nbeads
        q .+= selectdim(ρ,nd+1,i)*Z[i]
    end

    ϵ_r = external_field.ϵ_r
    P = external_field.plan
    iP = external_field.iplan
    map = external_field.map
    ψ = zeros(eltype(map),ngrid...)
    
    matmul!(ψ,P,q)
    elmul!(ψ,ψ,map)
    ψ[1] = 0.
    matmul!(ψ,iP,ψ)
    ψ = real.(ψ)
    # ψ .+= find_ψ_const(structure,external_field,model,ρ,z)

    Vext = zeros(Float64, ngrid..., nbeads)
    for i in 1:nbeads
        selectdim(Vext,nd+1,i) .= Z[i]*ψ 
    end

    return Vext ./ k_B / T
end

function find_ψ_const(structure::DFTStructure,external_field::ElectrostaticPotentialModel,model::ElectrolyteModel,ρ::Array{Float64},z)
    Z = model.charge
    nbeads = length(Z)
    nd = length(structure.ngrid)
    # obtain charge profiles
    ψ0 = 0.
    while true
        q = 0.
        dq = 0.
        for i in 1:nbeads
            q += sum(selectdim(ρ,nd+1,i)*Z[i])*exp(-Z[i]*ψ0)
            dq -= sum(selectdim(ρ,nd+1,i))*Z[i]^2*exp(-Z[i]*ψ0)
        end
        ψ0 -= q/dq
        # println("ψ0 = ", ψ0, " q = ", q)
        if abs(q) < 1e-6
            break
        end
    end
    return ψ0*k_B*structure.conditions[2]
end