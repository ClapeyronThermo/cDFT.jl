abstract type DFTStructure end
abstract type DFTStructureCart <: DFTStructure end
abstract type DFTStructureSphr <: DFTStructure end
abstract type DFTStructureCyl  <: DFTStructure end

abstract type DFTStructure1DCart <: DFTStructureCart end
abstract type DFTStructure2DCart <: DFTStructureCart end
abstract type DFTStructure3DCart <: DFTStructureCart end
abstract type DFTStructure1DSphr <: DFTStructureSphr end
abstract type DFTStructure1DCyl  <: DFTStructureCyl  end

dimension(x::DFTStructure) = dimension(typeof(x))
dimension(::Type{<:DFTStructure1DCart}) = 1
dimension(::Type{<:DFTStructure2DCart}) = 2
dimension(::Type{<:DFTStructure3DCart}) = 3
dimension(::Type{<:DFTStructure1DSphr}) = 1
dimension(::Type{<:DFTStructure1DCyl})  = 1

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

julia> structure = ExternalField1DCart((p, T), ρbulk, [0.5L, H-0.5L], 201, surface, H)
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
- `bounds`: `[lb, ub]`. `ub` is used as the aperture of the underlying quasi-discrete Hankel transform (QDHT); `lb` (which may be `0.0`) is used only to place an excluded-volume/wall external field (e.g. a spherical nanoparticle of radius `lb`), not to truncate the grid — the radial grid always spans from near `0` to `ub`.
- `ngrids`: The number of grid points used to represent the density profile.
- `transform`: The cached `Hankel.QDHT` used for all radial convolutions on this structure.
Example:
```julia
julia> structure = Uniform1DSphr((p, T), ρbulk, [0.0, 10L], 201)
```
"""
struct Uniform1DSphr{Q} <: DFTStructure1DSphr
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    bounds::Vector{Float64}
    ngrid::Tuple{Int64}
    transform::Q
end

function Uniform1DSphr(conditions,ρbulk,bounds::Vector{Float64},ngrid::Int64)
    Q = Hankel.QDHT(0, 2, bounds[2], ngrid)
    Uniform1DSphr(conditions,ρbulk,bounds,(ngrid,),Q)
end

"""
    Uniform1DCyl(conditions::Tuple{Float64,Float64}, ρbulk::Vector{Float64}, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate a uniform system in 1D-cylindrical coordinates (radial coordinate only; translationally invariant along the cylinder axis and rotationally symmetric about it). Contains:
- `conditions`: The p, T conditions of the system.
- `ρbulk`: The bulk density of each species in the system.
- `bounds`: `[lb, ub]`. `ub` is used as the aperture of the underlying quasi-discrete Hankel transform (QDHT); `lb` (which may be `0.0`, e.g. for fluid inside a pore) is used only to place an excluded-volume/wall external field (e.g. fluid outside a solid cylinder of radius `lb`), not to truncate the grid — the radial grid always spans from near `0` to `ub`.
- `ngrids`: The number of grid points used to represent the density profile.
- `transform`: The cached `Hankel.QDHT` used for all radial convolutions on this structure.
Example:
```julia
julia> structure = Uniform1DCyl((p, T), ρbulk, [0.0, 20L], 201)
```
"""
struct Uniform1DCyl{Q} <: DFTStructure1DCyl
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    bounds::Vector{Float64}
    ngrid::Tuple{Int64}
    transform::Q
end

function Uniform1DCyl(conditions,ρbulk,bounds::Vector{Float64},ngrid::Int64)
    Q = Hankel.QDHT(0, 1, bounds[2], ngrid)
    Uniform1DCyl(conditions,ρbulk,bounds,(ngrid,),Q)
end

radial_transform(structure::Union{Uniform1DSphr,Uniform1DCyl}) = structure.transform

"""
    RadialFrequency{FP,Q}

Wraps the `Hankel.QDHT` used by a `Uniform1DSphr`/`Uniform1DCyl` structure together with
the corresponding "ordinary frequency" vector `ω̄ = Q.k ./ 2π`. Using this convention
(rather than `Q.k` directly) means the *same* closed-form kernel formulas already used
for the Cartesian FFT path (e.g. `sin(ω̄R)/ω̄`-style expressions, with `R = 2π*width`)
can be reused unchanged for the radial case, just substituting this `ω̄` in place of
`sqrt.(sum(abs2,ω,dims=nd+1))`.

Returned by `structure_ω` for spherical/cylindrical structures in place of the raw
`Array{Complex}` returned for Cartesian ones; this lets `SWeightedDensity`/
`VWeightedDensity`/etc. dispatch to the correct (QDHT-based) constructor without any
change to the models' `get_fields` call sites, which just pass `ω` through positionally.
"""
struct RadialFrequency{FP<:AbstractFloat, Q<:Hankel.QDHT}
    Q::Q
    ω̄::Vector{FP}
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
    TwoPhase2DLamCart(conditions::Tuple{Float64,Float64}, ρbulk, ρbulk2, bounds::Matrix{Float64}, ngrid::Tuple{Int64,Int64})

The generic structure type used when trying to simulate a lamellar two-phase interface in 2D-cartesian coordinates (e.g. for a copolymer melt that microphase-separates into planar lamellae, or a slab-geometry vapour-liquid interface with a second transverse dimension). Contains:
- `conditions`: The p, T conditions at which the calculations are performed.
- `ρbulk`: The bulk density of each species in the first phase.
- `ρbulk2`: The bulk density of each species in the second phase.
- `bounds`: A 2×2 matrix specifying the bounds of the system along each dimension. The interface is located along the first dimension, with the second dimension left uniform/periodic.
- `ngrids`: The number of grid points used to represent the density profile along each dimension.

As with `TwoPhase1DCart`, the profiles are initialised as generic sigmoidals along the interface dimension, uniform along the other.
"""
struct TwoPhase2DLamCart <: DFTStructure2DCart
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    ρbulk2::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64}
end

"""
    TwoPhase3DLamCart(conditions::Tuple{Float64,Float64}, ρbulk, ρbulk2, bounds::Matrix{Float64}, ngrid::Tuple{Int64,Int64,Int64})

The generic structure type used when trying to simulate a lamellar two-phase interface in 3D-cartesian coordinates. Contains:
- `conditions`: The p, T conditions at which the calculations are performed.
- `ρbulk`: The bulk density of each species in the first phase.
- `ρbulk2`: The bulk density of each species in the second phase.
- `bounds`: A 3×2 matrix specifying the bounds of the system along each dimension. The interface is located along the first dimension, with the other two left uniform/periodic.
- `ngrids`: The number of grid points used to represent the density profile along each dimension.

As with `TwoPhase1DCart`, the profiles are initialised as generic sigmoidals along the interface dimension, uniform along the other two.
"""
struct TwoPhase3DLamCart <: DFTStructure3DCart
    conditions::Tuple{Float64,Float64}
    ρbulk::Vector{Float64}
    ρbulk2::Vector{Float64}
    bounds::Matrix{Float64}
    ngrid::Tuple{Int64,Int64,Int64}
end

"""
    TwoPhase2DHexCart(conditions::Tuple{Float64,Float64}, ρbulk, ρbulk2, bounds::Vector{Float64}, ngrid::Tuple{Int64})

The generic structure type used when trying to simulate a two-phase interface with hexagonal (cylindrical-domain) symmetry in a 2D-cartesian cross-section (e.g. a hexagonally-packed cylindrical copolymer microdomain viewed end-on). Contains:
- `conditions`: The p, T conditions at which the calculations are performed.
- `ρbulk`: The bulk density of each species in the first phase (domain center).
- `ρbulk2`: The bulk density of each species in the second phase (matrix).
- `bounds`: Specifies the (square) bounds of the 2D cross-section; the same `[lb, ub]` is applied to both dimensions.
- `ngrids`: The number of grid points used along each dimension (same for both).

The profile is initialised as a radially-symmetric sigmoidal (`tanh_prof`) centered on the cross-section.
"""
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
export Uniform1DSphr, Uniform1DCyl
export TwoPhase1DCart, TwoPhase2DLamCart, TwoPhase3DLamCart
export TwoPhase2DHexCart