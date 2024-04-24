abstract type BoundaryCondition end

"""
    FixedBoundary(value::Float64, direction::Int64)

A Fixed Boundary Condition where, when `converge!` updates the profiles, this boundary will remain unchanged. 

Inputs:
- `value::Float64`: The value of the boundary condition.
- `direction::Int64`: The direction of the boundary condition. If `direction == 1`, the boundary condition is applied to the right boundary. If `direction == -1`, the boundary condition is applied to the left boundary.
"""
struct FixedBoundary <: BoundaryCondition
    value::Float64
    direction::Int64
end

"""
    FreeBoundary(value::Float64, direction::Int64)

A Free Boundary Condition where, when `converge!` updates the profiles, this boundary is allowed to changed. 

Inputs:
- `value::Float64`: The value of the boundary condition.
- `direction::Int64`: The direction of the boundary condition. If `direction == 1`, the boundary condition is applied to the right boundary. If `direction == -1`, the boundary condition is applied to the left boundary.
"""
mutable struct FreeBoundary <: BoundaryCondition 
    value::Float64
    direction::Int64
end

"""
    PeriodicBoundary(direction::Int64)

A Periodic Boundary Condition where, when `converge!` updates the profiles.
"""
struct PeriodicBoundary <: BoundaryCondition 
    direction::Int64
end

"""
    get_boundary_conditions!(boundary_conditions::Tuple{BoundaryCondition,BoundaryCondition},ρ)

Given a tuple of boundary conditions, this function will return a 2x2 matrix of the boundary conditions. In the case of a free boundary, it will update the value.
"""
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
                boundary_conditions[i].value = ρ[end]
            elseif boundary_conditions[i].direction == -1
                bc[i,1]= ρ[1]
                bc[i,2]= ρ[1]
                boundary_conditions[i].value = ρ[1]
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