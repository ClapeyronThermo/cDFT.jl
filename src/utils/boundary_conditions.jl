abstract type BoundaryCondition end

struct FixedBoundary <: BoundaryCondition
    value::Float64
    direction::Int64
end

struct FreeBoundary <: BoundaryCondition 
    value::Float64
    direction::Int64
end

struct PeriodicBoundary <: BoundaryCondition 
    direction::Int64
end

function get_boundary_conditions!(boundary_conditions::Tuple{BoundaryCondition,BoundaryCondition},ρ)
    bc = zeros(Float64,2,2)
    for i in 1:length(boundary_conditions)
        if boundary_conditions[i] isa FixedBoundary
            bc[i,1]= boundary_conditions[i].value
            bc[i,2]= boundary_conditions[i].value
        elseif boundary_conditions[i] isa FreeBoundary
            if boundary_conditions[i].direction == 1
                bc[i,1]= ρ[end]
                bc[i,2]= ρ[end]
                boundary_conditions[i] = FreeBoundary(ρ[end], boundary_conditions[i].direction)
            elseif boundary_conditions[i].direction == -1
                bc[i,1]= ρ[1]
                bc[i,2]= ρ[1]
                boundary_conditions[i] = FreeBoundary(ρ[1], boundary_conditions[i].direction)
            end
        elseif boundary_conditions[i] isa PeriodicBoundary
            if boundary_conditions[i].direction == 1
                bc[i,1] = ρ[1]
                bc[i,2] = ρ[2]
            elseif boundary_conditions[i].direction == -1
                bc[i,1] = ρ[end-1]
                bc[i,2] = ρ[end]
            end
        else
            error("Boundary condition not recognized")
        end
    end
    return bc
end