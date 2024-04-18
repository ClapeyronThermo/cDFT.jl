abstract type BoundaryCondition end

struct FixedBoundary <: BoundaryCondition
    value::Float64
end

abstract type PeriodicBoundary <: BoundaryCondition end

abstract type MirrorBoundary <: BoundaryCondition end