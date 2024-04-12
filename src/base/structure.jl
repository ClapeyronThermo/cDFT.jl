abstract type DFTStructure end 
abstract type DFTStructure1D <: DFTStructure end
abstract type DFTStructure1DCart <: DFTStructure1D end
abstract type DFTStructure1DSpherical <: DFTStructure1D end

abstract type DFTStructure2D <: DFTStructure end
abstract type DFTStructure3D <: DFTStructure end

struct SurfaceTension1DCart <: DFTStructure1DCart 
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Tuple{Float64,Float64}
    ngrid::Int64
end

struct Uniform1DCart <: DFTStructure1DCart
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Tuple{Float64,Float64}
    ngrid::Int64
end