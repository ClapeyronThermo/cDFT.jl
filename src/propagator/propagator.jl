struct IdealPropagator <: DFTPropagator end
struct TangentHSPropagator <: DFTPropagator end

include("ideal.jl")
include("tangent_hs.jl")