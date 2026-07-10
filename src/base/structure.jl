abstract type DFTStructure{Dim,Coord,Type} end
abstract type DFTCoordType end
abstract type DFTStructureType end

struct Structure{Dim,Coord,Type} <: DFTStructure{Dim,Coord,Type}
    conditions::Tuple{Float64,Float64}
    bounds::NTuple{Dim,NTuple{2,Float64}}
    ρbulk::Vector{Float64}
    ngrid::NTuple{Dim,Int64}
    system_type::Type
end

function Structure{Dim,Coord}(conditions,bounds,ρbulk,ngrid,system_type)
    assert_ngrid(ngrid)
    assert_bounds(bounds)
    norm_bounds = normalize_bounds(bounds,Val(Dim))
    norm_ngrid = normalize_ngrid(bounds,Val(Dim))
    return Structure{Dim,Coord,typeof(system_type)}(conditions,bounds,ρbulk,ngrid,system_type)
end

struct Cartesian <: DFTCoordType end
const Cart = Cartesian()

struct Cylindrical <: DFTCoordType end
const Cyl = Cylindrical()

struct Spherical <: DFTCoordType end
const Sphr = Spherical()

#a way to check the struct first: DFTStructByCoord{Cartesian} = DFTStructure{N,Cartesian}
const DFTStructByCoord{T,N,S} = DFTStructure{N,T,S} where {T,N,S}
const DFTStructByType{S,N,T} = DFTStructure{N,T,S} where {S,N,T}

dimension(x::DFTStructure{N}) where N = N
dimension(::Type{<:DFTStructure{N}}) where N = N

bounds(structure::DFTStructure,dim::Int) = structure.bounds[dim]
ngrid(structure::DFTStructure,dim) =  structure.ngrid[dim]
structure_type_data(structure::DFTStructure) = structure.system_type

function uniform_range(structure::DFTStructure,dim::Int)
    lb,ub = bounds(structure,dim)
    return LinRange(lb,ub,ngrid(structure,dim))
end

uniform_range(structure::DFTStructure) = uniform_range(structure,1)

#=
utils
=#

assert_ngrid(::Number, ::Val{1}) = nothing
assert_ngrid(::Number, ::Val{N}) where N = throw(DimensionMismatch("Expected 1‑D grid, got scalar"))
assert_ngrid(::NTuple{D}, ::Val{D}) where D = nothing
assert_ngrid(::Tuple, ::Val{D}) where D = throw(DimensionMismatch("Tuple length does not match $D"))

function assert_ngrid(arr::AbstractArray, ::Val{D}) where D
    (length(arr) == D) || throw(DimensionMismatch("Array length does not match $D"))
    return nothing
end

assert_ngrid(::Any, ::Val) = throw(DimensionMismatch("Invalid grid type"))

normalize_ngrid(x::Number, ::Val{1}) = (x,)
normalize_ngrid(::Number, ::Val{N}) where N = throw(DimensionMismatch("Scalar cannot be $N‑D grid"))
normalize_ngrid(t::NTuple{D}, ::Val{D}) where D = t
normalize_ngrid(::Tuple, ::Val{D}) where D = throw(DimensionMismatch("Tuple length does not match $D"))

function normalize_ngrid(arr::AbstractArray, ::Val{D}) where D
    length(arr) == D || throw(DimensionMismatch("Array length does not match $D"))
    return Tuple(arr)
end
normalize_ngrid(::Any, ::Val) = throw(DimensionMismatch("Invalid grid type"))

function assert_bounds(t::Tuple, ::Val{N}) where N
    length(t) == N || throw(DimensionMismatch("Tuple length $(length(t)) != $N"))
    for (i, x) in enumerate(t)
        if !(x isa Tuple && length(x) == 2 && x[1] isa Number && x[2] isa Number)
            throw(DimensionMismatch("Element $i is not a 2‑tuple of numbers"))
        end
    end
    return nothing
end

assert_bounds(v::AbstractVector, ::Val{1}) =
    (length(v) == 2 && all(isa(x, Number) for x in v)) ||
    throw(DimensionMismatch("Vector must have exactly 2 numbers for 1‑D bound"))

assert_bounds(::AbstractVector, ::Val{N}) where N =
    throw(DimensionMismatch("Vector is only valid for 1‑D bound, got $N"))

function assert_bounds(m::AbstractMatrix, ::Val{N}) where N
    size(m) == (N, 2) || throw(DimensionMismatch("Matrix size $(size(m)) != ($N, 2)"))
    all(isa(x, Number) for x in m) ||
        throw(DimensionMismatch("Matrix contains non‑numeric values"))
    return nothing
end

assert_bounds(::Any, ::Val) =  throw(DimensionMismatch("Invalid bounds type"))

normalize_bounds(t::NTuple{N,NTuple{2,Float64}}, ::Val{N}) where N = t
normalize_bounds(v::AbstractVector, ::Val{1}) = ((v[1], v[2]))
normalize_bounds(m::AbstractMatrix, ::Val{N}) where N = ntuple(i -> ([i, 1],[i, 2]),Val(N))
normalize_bounds(::Any, ::Val) = throw(DimensionMismatch("Invalid bounds type"))

function _1d_to_2d(_bounds,_ngrid)
    assert_bounds(_bounds,Val(1))
    assert_ngrid(_ngrid,Val(1))
    assert_1d(_bounds,_ngrid)
    lb,ub = _bounds
    n = only(_ngrid)
    _new_bounds = ((lb ub),(lb ub))
    _new_ngrid = (n,n)
    return _new_bounds,_new_ngrid
end

function _1d_to_3d(_bounds,_ngrid)
    assert_bounds(_bounds,Val(1))
    assert_ngrid(_ngrid,Val(1))
    lb,ub = _bounds
    n = only(_ngrid)
    _new_bounds = ((lb ub),(lb ub),(lb ub))
    _new_ngrid = (n,n,n)
    return _new_bounds,_new_ngrid
end

struct ExternalField{E<: ExternalFieldModel} <: DFTStructureType
    external_field::E
    width::Float64
end

"""
    ExternalField1DCart(conditions::Tuple{Float64,Float64}, ρbulk::Vector{Float64}, bounds::Vector{Float64}, ngrid::Int64, external_field::ExternalFieldModel, width::Float64)

The generic structure type used when trying to simulate solid-fluid interfaces in 1D-cartesian coordinates. Its inputs are:
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
function ExternalField1DCart(conditions,ρbulk,bounds,ngrid,external_field,width)
    ext = ExternalField(external_field,width)
    return Structure{1,Cartesian}(conditions,ρbulk,bounds,ngrid,ext)
end

struct UniformGrid{T} <: DFTStructureType
    transform::T
end

"""
    Uniform1DCart(conditions::Tuple{Float64,Float64}, ρbulk, bounds, ngrid)

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
function Uniform1DCart(conditions,ρbulk,bounds,ngrid)
    return Structure{1,Cartesian}(conditions,ρbulk,bounds,ngrid,UniformGrid(nothing))
end


"""
    Uniform2DCart(conditions, ρbulk, bounds, ngrid::Int64)

The generic structure type used when trying to simulate a uniform system in 2D-cartesian coordinates. Its inputs are::
- `conditions`: The p, T conditions of the system.
- `ρbulk`: The bulk density of each species in the system.
- `bounds`: A 2×2 matrix specifying the bounds of the system along each dimension.
- `ngrids`: The number of grid points used to represent the density profile along each dimension.

This structure should generally be used to benchmark the DFT code against the bulk calculations.

## Example:
```julia
julia> structure = Uniform2DCart((p, T), ρbulk, [-10L 10L; -10L 10L], 101)
```
"""
Uniform2DCart(conditions, ρbulk, bounds, ngrid) = Structure{2,Cartesian}(conditions,ρbulk,bounds,ngrid,UniformGrid())

"""
    Uniform3DCart(conditions, ρbulk, bounds, ngrid)

The generic structure type used when trying to simulate a uniform system in 3D-cartesian coordinates. Its inputs are::
- `conditions`: The p, T conditions of the system.
- `ρbulk`: The bulk density of each species in the system.
- `bounds`: A 3×2 matrix specifying the bounds of the system along each dimension.
- `ngrids`: The number of grid points used to represent the density profile along each dimension.

This structure should generally be used to benchmark the DFT code against the bulk calculations.

## Example:
```julia
julia> structure = Uniform3DCart((p, T), ρbulk, [-10L 10L; -10L 10L; -10L 10L], 51)
```
"""
Uniform3DCart(conditions, ρbulk, bounds, ngrid) = Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,UniformGrid())

#=================

Radial transforms over uniform grids

=================#

radial_transform(structure::DFTStructure{<:Any,<:Any,<:UniformGrid}) = radial_transform(structure_type_data(structure))
radial_transform(data::UniformGrid{Q}) where Q = data.transform

#generators: from uniform grid to the adequate radial transform

function to_radial(s0::DFTStructure{1,Spherical,UniformGrid{Nothing}})
    _,ub = bounds(s0)
    grid = UniformGrid(Hankel.QDHT(0, 2, ub, ngrid(s0)))
    Structure{1,Spherical}(s0.conditions,s0.ρbulk,s0.bounds,s0.ngrid,grid)
end

function to_radial(s0::DFTStructure{1,Cylindrical,UniformGrid{Nothing}})
    _,ub = bounds(s0)
    grid = UniformGrid(Hankel.QDHT(0, 1, ub, ngrid(s0)))
    Structure{1,Cylindrical}(s0.conditions,s0.ρbulk,s0.bounds,s0.ngrid,grid)
end

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
function Uniform1DSphr(conditions,ρbulk,bounds,ngrid)
    to_radial(Structure{1,Spherical}(conditions,ρbulk,bounds,ngrid,UniformGrid()))
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
function Uniform1DCyl(conditions,ρbulk,bounds,ngrid)
    to_radial(Structure{1,Cylindrical}(conditions,ρbulk,bounds,ngrid,UniformGrid()))
end

#=================

Two Phase Systems

The free parameter is a symbol, specifying the type of system of the second phase.
ej: TwoPhaseSystem{:Cartesian}

available symbols are:

- :Cartesian
- :Lamellar
- :HexLattice
- :Spherical
=================#

struct TwoPhaseSystem{T} <: DFTStructureType
    ρbulk2::Vector{Float64}
end

"""
    TwoPhase1DCart(conditions::Tuple{Float64,Float64}, ρbulk, ρbulk2, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate two-phase interfaces in 1D-cartesian coordinates. Its inputs are:
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
function TwoPhase1DCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    return Structure{1,Cartesian}(conditions,ρbulk,bounds,ngrid,TwoPhaseSystem{:Cartesian}(ρbulk2))
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
function TwoPhase2DLamCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    return Structure{2,Cartesian}(conditions,ρbulk,bounds,ngrid,TwoPhaseSystem{:Lamellar}(ρbulk2))
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
function TwoPhase3DLamCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    return Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,TwoPhaseSystem{:Lamellar}(ρbulk2))
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
function TwoPhase2DHexCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    bounds_2d,ngrid_2d = _1d_to_2d(_bounds,_ngrid)
    return Structure{2,Cartesian}(conditions,ρbulk,bounds_2d,ngrid_2d,TwoPhaseSystem{:HexLattice}(ρbulk2))
end

"""
    TwoPhase3DHexCart(conditions::Tuple{Float64,Float64}, ρbulk, ρbulk2, bounds::Vector{Float64}, ngrid::Tuple{Int64})

The generic structure type used when trying to simulate a two-phase interface with hexagonal (cylindrical-domain) symmetry in 3D-cartesian coordinates (e.g. a hexagonally-packed array of cylindrical copolymer microdomains, extruded along the third, translationally-invariant dimension). Contains:
- `conditions`: The p, T conditions at which the calculations are performed.
- `ρbulk`: The bulk density of each species in the first phase (domain center).
- `ρbulk2`: The bulk density of each species in the second phase (matrix).
- `bounds`: Specifies the (cubic) bounds of the system; the same `[lb, ub]` is applied to all three dimensions.
- `ngrids`: The number of grid points used along each dimension (same for all three).

The profile is initialised as a radially-symmetric sigmoidal (`tanh_prof`) in the cross-section spanned by the first two dimensions, uniform along the third.
"""
function TwoPhase3DHexCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    bounds_3d,ngrid_3d = _1d_to_3d(_bounds,_ngrid)
    Structure{3,Cartesian}(conditions,ρbulk,bounds_3d,ngrid_3d,TwoPhaseSystem{:HexLattice}(ρbulk2))
end

"""
    TwoPhase3DSphrCart(conditions::Tuple{Float64,Float64}, ρbulk, ρbulk2, bounds::Vector{Float64}, ngrid::Tuple{Int64})

The generic structure type used when trying to simulate a two-phase interface with spherical symmetry (e.g. a droplet or bubble) embedded in a 3D-cartesian box. Contains:
- `conditions`: The p, T conditions at which the calculations are performed.
- `ρbulk`: The bulk density of each species in the first phase (droplet/bubble interior).
- `ρbulk2`: The bulk density of each species in the second phase (surrounding bulk).
- `bounds`: Specifies the (cubic) bounds of the system; the same `[lb, ub]` is applied to all three dimensions.
- `ngrids`: The number of grid points used along each dimension (same for all three).

The profile is initialised as a radially-symmetric sigmoidal (`tanh_prof`) centered on the box, i.e. a sphere of the first phase surrounded by the second.
"""
function TwoPhase3DSphrCart(conditions,ρbulk,ρbulk2,bounds,ngrid)
    bounds_3d,ngrid_3d = _1d_to_3d(_bounds,_ngrid)
    Structure{3,Cartesian}(conditions,ρbulk,bounds_2d,ngrid_2d,TwoPhaseSystem{:Spherical}(ρbulk2))
end

# ── Block-copolymer microphase morphologies ─────────────────────────────────
#
# Unlike the TwoPhaseSystem structure above (one scalar profile per *component*,
# transitioning between two bulk phases), these represent a single periodic unit cell of
# a microphase-separated block-copolymer melt, in which *different groups within the same
# component* (e.g. "A" vs "B" from a `custom_structure`) enrich in different spatial
# domains. `core_groups` names which groups form the minority/core domain (spheres for
# BCC, cylinders for Hex, the network for Gyroid, one set of layers for Lamellar); every
# other group in the model is treated as the surrounding matrix. Requires a
# group-contribution model (anything with `model.groups`, e.g. `HeterogcPCPSAFT`) — see
# `src/structure/morphology.jl` for the seed-profile math.


struct BlockCopolymerMorphology{T} <: DFTStructureType end
    core_groups::Vector{String}
    amplitude::Float64
    periods::Int

    function BlockCopolymerMorphology{T}(core_groups::Vector{String},amplitude::Float64,periods::Int) where T
        periods >= 1 || throw(ArgumentError("periods must be a positive integer, got $periods"))
        return new{T}(core_groups,amplitude,periods)
    end
end

"""
    LamellarStack1DCart(conditions, ρbulk, bounds::Vector{Float64}, ngrid::Int64; core_groups, amplitude=1.0, periods=1)

Seeds a periodic lamellar (alternating-layer) block-copolymer morphology in 1D-cartesian
coordinates: `core_groups` enrich in one set of layers, every other group in the
alternating layers. `periods` (a positive integer, default `1`) sets how many full
lamellar periods are seeded across the box (`bounds`) — `1` means the period equals the
full box length.
"""
function LamellarStack1DCart(conditions,ρbulk,bounds,ngrid; core_groups, amplitude=0.3, periods=1)
    lam = BlockCopolymerMorphology{:Lamellar}(core_groups,amplitude,periods)
    Structure{1,Cartesian}(conditions,ρbulk,bounds,ngrid,lam)
end

"""
    LamellarStack2DCart(conditions, ρbulk, bounds::Matrix{Float64}, ngrid::Tuple{Int64,Int64}; core_groups, amplitude=1.0, periods=1)

2D-cartesian counterpart of [`LamellarStack1DCart`](@ref cDFT.LamellarStack1DCart): layers
alternate along the first dimension, uniform along the second. `periods` sets how many
lamellar periods are seeded along the first dimension.
"""
function LamellarStack2DCart(conditions,ρbulk,bounds,ngrid; core_groups, amplitude=0.3, periods=1)
    lam = BlockCopolymerMorphology{:Lamellar}(core_groups,amplitude,periods)
    Structure{2,Cartesian}(conditions,ρbulk,bounds,ngrid,lam)
end

"""
    LamellarStack3DCart(conditions, ρbulk, bounds::Matrix{Float64}, ngrid::Tuple{Int64,Int64,Int64}; core_groups, amplitude=1.0, periods=1)

3D-cartesian counterpart of [`LamellarStack1DCart`](@ref cDFT.LamellarStack1DCart): layers
alternate along the first dimension, uniform along the other two. `periods` sets how many
lamellar periods are seeded along the first dimension.
"""
function LamellarStack3DCart(conditions,ρbulk,bounds,ngrid; core_groups, amplitude=0.3, periods=1)
    lam = BlockCopolymerMorphology{:Lamellar}(core_groups,amplitude,periods)
    Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,lam)
end

HexLattice(core_groups,amplitude,periods) = BlockCopolymerMorphology{:HexLattice}(core_groups,amplitude,periods)

function assert_hex_bounds(bounds,::Val{N}) where N
    assert_bounds(bounds,Val(N))
    std_bounds = normalize_bounds(bound,Val(N))
    L = ntuple(i -> std_bounds[i][2] - std_bounds[i][1],Val(N))
    Lx,Ly = L[1],L[2]
    isapprox(Ly, sqrt(3)*Lx; rtol=0.05) || @warn "HexLattice$(N)DCart: Ly = (uby - lby) should span ≈ √3 × Lx = (ubx - lbx) for a clean hexagonal supercell (got Ly/Lx = $(Ly/Lx)); the initial guess will be distorted."
end

"""
    HexLattice2DCart(conditions, ρbulk, bounds::Matrix{Float64}, ngrid::Tuple{Int64,Int64}; core_groups, amplitude=1.0, periods=1)

Seeds a periodic hexagonally-packed-cylinder block-copolymer morphology (2D cross-section):
`core_groups` enrich into two cylindrical domains per unit cell, matching the standard
rectangular 2-cylinder supercell that reproduces a true hexagonal lattice under periodic
(FFT) boundary conditions on a Cartesian grid. `bounds`' second dimension should span
`√3 ×` the first (a warning is emitted otherwise — this only affects the quality of the
initial guess, not the correctness of a subsequently converged/evolved profile). `periods`
(a positive integer, default `1`) tiles this 2-cylinder supercell `periods` times along
each dimension.
"""
function HexLattice2DCart(conditions,ρbulk,bounds,ngrid; core_groups, amplitude=0.3, periods=1)
    hex = HexLattice(core_groups,amplitude,periods)
    assert_hex_bounds(bounds,Val(2))
    Structure{2,Cartesian}(conditions,ρbulk,bounds,ngrid,hex)
end

"""
    HexLattice3DCart(conditions, ρbulk, bounds::Matrix{Float64}, ngrid::Tuple{Int64,Int64,Int64}; core_groups, amplitude=1.0, periods=1)

3D-cartesian counterpart of [`HexLattice2DCart`](@ref cDFT.HexLattice2DCart): the hexagonal
cylinder lattice is extruded, uniform, along the third dimension.
"""
function HexLattice3DCart(conditions,ρbulk,bounds,ngrid; core_groups, amplitude=0.3, periods=1)
    hex = HexLattice(core_groups,amplitude,periods)
    assert_hex_bounds(bounds,Val(3))
    Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,hex)
end

function assert_cubic_bounds(bounds)
    assert_bounds(bounds,Val(3))
    std_bounds = normalize_bounds(bound,Val(3))
    L = ntuple(i -> std_bounds[i],[2] - std_bounds[i][1],Val(3))
    isapprox(L[1],L[2];rtol=0.02) && isapprox(L[2],L[3];rtol=0.02) || @warn "Cubic Bounds Check: bounds should be cubic (equal extent in all 3 dimensions) for a clean unit cell; got extents $(L)."
end

"""
    BCC3DCart(conditions, ρbulk, bounds::Matrix{Float64}, ngrid::Tuple{Int64,Int64,Int64}; core_groups, amplitude=1.0, periods=1)

Seeds a periodic body-centered-cubic (BCC, Im-3m) sphere block-copolymer morphology:
`core_groups` enrich into spheres at the corners and body-center of the cubic unit cell
(`bounds`, expected cubic — a warning is emitted otherwise). `periods` (a positive
integer, default `1`) tiles `periods` unit cells along each dimension.
"""
function BCC3DCart(conditions,ρbulk,bounds,ngrid; core_groups, amplitude=0.3, periods=1)
    bcc = BlockCopolymerMorphology{:BodyCenteredCubic}(core_groups,amplitude,periods)
    assert_cubic_bounds(bounds)
    Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,bcc)
end

"""
    Gyroid3DCart(conditions, ρbulk, bounds::Matrix{Float64}, ngrid::Tuple{Int64,Int64,Int64}; core_groups, amplitude=1.0, periods=1)

Seeds a periodic gyroid (Ia-3d) block-copolymer morphology, using the standard Schoen
gyroid level-set as the initial guess: `core_groups` enrich on one side of the level set
(one of the two mutually-interpenetrating networks), the rest of the model on the other.
`bounds` is expected cubic (a warning is emitted otherwise). `periods` (a positive
integer, default `1`) tiles `periods` unit cells along each dimension.
"""
function Gyroid3DCart(conditions,ρbulk,bounds,ngrid; core_groups, amplitude=0.3, periods=1)
    gyr = BlockCopolymerMorphology{:Gyroid}(core_groups,amplitude,periods)
    assert_cubic_bounds(bounds)
    Structure{3,Cartesian}(conditions,ρbulk,bounds,ngrid,gyr)
end

export Structure, Cartesian, Cylindrical, Spherical
export UniformGrid, TwoPhaseSystem, BlockCopolymerMorphology
export Uniform1DCart, ExternalField1DCart
export Uniform1DSphr, Uniform1DCyl
export TwoPhase1DCart, TwoPhase2DLamCart, TwoPhase3DLamCart
export TwoPhase2DHexCart
export LamellarStack1DCart, LamellarStack2DCart, LamellarStack3DCart
export HexLattice2DCart, HexLattice3DCart
export BCC3DCart, Gyroid3DCart