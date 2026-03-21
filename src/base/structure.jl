abstract type DFTStructure end 
abstract type DFTStructure1D <: DFTStructure end
abstract type DFTStructure1DCart <: DFTStructure1D end
abstract type DFTStructure1DSphr <: DFTStructure1D end

abstract type DFTStructure2D <: DFTStructure end
abstract type DFTStructure2DCart <: DFTStructure2D end
abstract type DFTStructure3D <: DFTStructure end
abstract type DFTStructure3DCart <: DFTStructure3D end

dimension(x::DFTStructure) = dimension(typeof(x))
dimension(::Type{<:DFTStructure1D}) = 1
dimension(::Type{<:DFTStructure2D}) = 2
dimension(::Type{<:DFTStructure3D}) = 3

struct DFTBounds{N}
    lb::NTuple{N,Float64}
    ub::NTuple{N,Float64}
end

function DFTBounds{1}(x::Vector{Float64})
    DFTBounds((x[1],),(x[2],))
end

function DFTBounds{2}(x::Matrix{Float64})
    DFTBounds((x[1,1],x[2,1]),(x[1,2],x[2,2]))
end

function DFTBounds{3}(x::Matrix{Float64})
    DFTBounds((x[1,1],x[2,1],x[3,1]),(x[1,2],x[2,2],x[3,2]))
end

function uniform_range(structure::DFTStructure,dim::Int)
    bounds = DFTBounds{dimension((structure))}(structure.bounds)
    lb,ub = bounds.lb,bounds.ub
    grid = structure.ngrid
    return LinRange(lb[dim],ub[dim],grid[dim])
end

function bounds(structure::DFTStructure,dim::Int)
    _bounds = DFTBounds{dimension((structure))}(structure.bounds)
    lb,ub = _bounds.lb,_bounds.ub
    return lb[dim], ub[dim]  
end

uniform_range(structure::DFTStructure) = uniform_range(structure,1)

"""
    ExternalField1DCart(conditions::Tuple{Float64,Float64}, ρbulk::Vector{Float64}, bounds::Vector{Float64}, ngrid::Int64, external_field::ExternalFieldModel, width::Float64)

The generic structure type used when trying to simulate solid-fluid interfaces in 1D-cartesian coordinates. Contains:
- `conditions`: The p, T conditions of the total system before it splits between the two liquid phases.
- `ρbulk`: The bulk density of each species in the system.
- `bounds`: Specifies the location of the bounds of the system. The interface will be located in the middle.
- `ngrids`: The number of grid points used to represent the density profile.
- `external_field`: The external field model used to calculate the external field.
- `width`: The surface-to-surface separation.

Example:
```julia
julia> model = PCSAFT(["carbon dioxide"])

julia> ρbulk = [molar_density(model,1e5,298.15)]

julia> L = length_scale(model)

julia> H = 10L

julia> surface = Steele(["graphite"])

julia> structure = InterfacialTension1DCart((p, T), ρbulk, [0.5L, H-0.5L], 201, surface, H)
```
"""
struct ExternalField1DCart <: DFTStructure1DCart 
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    bounds::Vector{Float64}
    ngrid::Tuple{Int64}
    external_field::ExternalFieldModel
    width::Float64
end

function ExternalField1DCart(conditions,ρbulk,bounds,ngrid::Int64,external_field,width)
    ExternalField1DCart(conditions,bounds,(ngrid,),external_field,width)
end

"""
    Uniform1DCart(conditions::Tuple{Float64,Float64}, ρbulk, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate a uniform system in 1D-cartesian coordinates. Contains:
- `conditions`: The p, T conditions of the system.
- `ρbulk`: The bulk density of each species in the system.
- `bounds`: Specifies the location of the bounds of the system.
- `ngrids`: The number of grid points used to represent the density profile.
This structure should generally be used to benchmark the DFT code against the bulk calculations.
Example:
```julia
julia> structure = Uniform1DCart((p, T) , ρbulk, [-10L, 10L], 201)
```
"""
struct Uniform1DCart <: DFTStructure1DCart
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    bounds::Vector{Float64}
    ngrid::Tuple{Int64}
end

function Uniform1DCart(conditions,ρbulk,bounds,ngrid::Int64)
    Uniform1DCart(conditions,ρbulk,bounds,(ngrid,))
end

"""
    Uniform1DCart(conditions::Tuple{Float64,Float64}, ρbulk, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate a uniform system in 1D-cartesian coordinates. Contains:
- `conditions`: The p, T conditions of the system.
- `ρbulk`: The bulk density of each species in the system.
- `bounds`: Specifies the location of the bounds of the system.
- `ngrids`: The number of grid points used to represent the density profile.
This structure should generally be used to benchmark the DFT code against the bulk calculations.
Example:
```julia
julia> structure = Uniform1DCart((p, T) , ρbulk, [-10L, 10L], 201)
```
"""
struct Uniform3DCart <: DFTStructure3DCart
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64,Int64}
end

function Uniform3DCart(conditions,ρbulk,bounds,ngrid::Int64)
    Uniform3DCart(conditions,ρbulk,bounds,(ngrid,ngrid,ngrid))
end

"""
    Uniform1DCart(conditions::Tuple{Float64,Float64}, ρbulk, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate a uniform system in 1D-cartesian coordinates. Contains:
- `conditions`: The p, T conditions of the system.
- `ρbulk`: The bulk density of each species in the system.
- `bounds`: Specifies the location of the bounds of the system.
- `ngrids`: The number of grid points used to represent the density profile.
This structure should generally be used to benchmark the DFT code against the bulk calculations.
Example:
```julia
julia> structure = Uniform1DCart((p, T) , ρbulk, [-10L, 10L], 201)
```
"""
struct Uniform2DCart <: DFTStructure2DCart
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64}
end

function Uniform2DCart(conditions,ρbulk,bounds,ngrid::Int64)
    Uniform2DCart(conditions,ρbulk,bounds,(ngrid,ngrid))
end

"""
    Uniform1DSphr(conditions::Tuple{Float64,Float64}, ρbulk::Vector{Float64}, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate a uniform system in 1D-spherical coordinates. Contains:
- `conditions`: The p, T conditions of the system.
- `ρbulk`: The bulk density of each species in the system.
- `bounds`: Specifies the location of the bounds of the system.
- `ngrids`: The number of grid points used to represent the density profile.
This structure should generally be used to benchmark the DFT code against the bulk calculations. As this is spherical coordinates, the lower bound must be greater than 0.
Example:
```julia
julia> structure = Uniform1DSphr((p, T), ρbulk, [0L, 10L], 201)
```
"""
struct Uniform1DSphr <: DFTStructure1DSphr
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Tuple{Int64}
end

function Uniform1DSphr(conditions,ρbulk,ρbulk2,bounds,ngrid::Int64)
    Uniform1DSphr(conditions,bounds,(ngrid,))
end

"""
    TwoPhase1DCart(conditions::Tuple{Float64,Float64}, ρbulk, ρbulk2, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate two-phase interfaces in 1D-cartesian coordinates. Contains:
- `conditions`: The p, T conditions at which the calculations are performed.
- `ρbulk`: The bulk density of each species in the first phase.
- `ρbulk2`: The bulk density of each species in the second phase.
- `bounds`: Specifies the location of the bounds of the system. The interface will be located in the middle.
- `ngrids`: The number of grid points used to represent the density profile.
In the case of the surface tension calculation, the pressure specified must be the saturation pressure at T and x. As the density profiles in the liquid and vapour phases are expected to reach their bulk values, the densities at the bounds are _fixed_ at their bulk values. In general, it is recommends to use a width of about `20L` (`L` being the length scale of the model) and about 201 grid points.

The profiles will be initialised as generic sigmoidals of the form:
`tanh_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*tanh((x-shift)*coef)+1/2*(start+stop)`

Example:
```julia
julia> model = PCSAFT(["water"])

julia> T = 298.15

julia> (p, vl, vv) = saturation_pressure(model, T)

julia> ρ1 = [1.0]./vl

julia> ρ2 = [1.0]./vv

julia> L = length_scale(model)

julia> structure = TwoPhase1DCart((p, T), ρ1, ρ2, [-10L, 10L], 201)
```
"""
struct TwoPhase1DCart <: DFTStructure1DCart 
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    ρbulk2::Vector{Float64}
    bounds::Vector{Float64}
    ngrid::Tuple{Int64}
end

function TwoPhase1DCart(conditions,ρbulk,ρbulk2,bounds,ngrid::Int64)
    TwoPhase1DCart(conditions,ρbulk,ρbulk2,bounds,(ngrid,))
end


"""
    TwoPhase1DCart(conditions::Tuple{Float64,Float64}, ρbulk, ρbulk2, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate two-phase interfaces in 1D-cartesian coordinates. Contains:
- `conditions`: The p, T conditions at which the calculations are performed.
- `ρbulk`: The bulk density of each species in the first phase.
- `ρbulk2`: The bulk density of each species in the second phase.
- `bounds`: Specifies the location of the bounds of the system. The interface will be located in the middle.
- `ngrids`: The number of grid points used to represent the density profile.
In the case of the surface tension calculation, the pressure specified must be the saturation pressure at T and x. As the density profiles in the liquid and vapour phases are expected to reach their bulk values, the densities at the bounds are _fixed_ at their bulk values. In general, it is recommends to use a width of about `20L` (`L` being the length scale of the model) and about 201 grid points.

The profiles will be initialised as generic sigmoidals of the form:
`tanh_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*tanh((x-shift)*coef)+1/2*(start+stop)`

Example:
```julia
julia> model = PCSAFT(["water"])

julia> T = 298.15

julia> (p, vl, vv) = saturation_pressure(model, T)

julia> ρ1 = [1.0]./vl

julia> ρ2 = [1.0]./vv

julia> L = length_scale(model)

julia> structure = TwoPhase1DCart((p, T), ρ1, ρ2, [-10L, 10L], 201)
```
"""
struct TwoPhase2DLamCart <: DFTStructure2DCart 
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    ρbulk2::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64}
end

struct TwoPhase3DLamCart <: DFTStructure3DCart 
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    ρbulk2::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64,Int64}
end

struct TwoPhase2DHexCart <: DFTStructure2DCart 
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    ρbulk2::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64}
end

function TwoPhase2DHexCart(conditions,ρbulk,ρbulk2,bounds::Vector{Float64},ngrid::Tuple{Int64})
    TwoPhase2DHexCart(conditions,ρbulk,ρbulk2,[bounds[1] bounds[2]; bounds[1] bounds[2]],(ngrid[1],ngrid[1]))
end

struct TwoPhase3DHexCart <: DFTStructure3DCart 
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    ρbulk2::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64,Int64}
end

function TwoPhase3DHexCart(conditions,ρbulk,ρbulk2,bounds::Vector{Float64},ngrid::Tuple{Int64})
    TwoPhase3DHexCart(conditions,ρbulk,ρbulk2,[bounds[1] bounds[2]; bounds[1] bounds[2]; bounds[1] bounds[2]],(ngrid[1],ngrid[1],ngrid[1]))
end

struct TwoPhase3DSphrCart <: DFTStructure3DCart 
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    ρbulk2::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64,Int64}
end

function TwoPhase3DSphrCart(conditions,ρbulk,ρbulk2,bounds::Vector{Float64},ngrid::Tuple{Int64})
    TwoPhase3DSphrCart(conditions,ρbulk,ρbulk2,[bounds[1] bounds[2]; bounds[1] bounds[2]; bounds[1] bounds[2]],(ngrid[1],ngrid[1],ngrid[1]))
end

export Uniform1DCart, ExternalField1DCart
export Uniform1DSphr
export TwoPhase1DCart, TwoPhase2DLamCart, TwoPhase3DLamCart
export TwoPhase2DHexCart