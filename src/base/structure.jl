abstract type DFTStructure end 
abstract type DFTStructure1D <: DFTStructure end
abstract type DFTStructure1DCart <: DFTStructure1D end
abstract type DFTStructure1DSphr <: DFTStructure1D end

abstract type DFTStructure2D <: DFTStructure end
abstract type DFTStructure3D <: DFTStructure end

struct SurfaceTension1DCart <: DFTStructure1DCart 
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
end

struct InterfacialTension1DCart <: DFTStructure1DCart 
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
end

struct Uniform1DCart <: DFTStructure1DCart
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
end

struct Uniform1DSphr <: DFTStructure1DSphr
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
end

struct InterfacialTension1DSphr <: DFTStructure1DSphr
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
    core_composition::Vector{Float64}
    r_interface::Float64
end

export SurfaceTension1DCart, InterfacialTension1DCart, Uniform1DCart
export Uniform1DSphr, InterfacialTension1DSphr