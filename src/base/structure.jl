abstract type DFTStructure end 
abstract type DFTStructure1D <: DFTStructure end
abstract type DFTStructure1DCart <: DFTStructure1D end
abstract type DFTStructure1DSphr <: DFTStructure1D end

abstract type DFTStructure2D <: DFTStructure end
abstract type DFTStructure2DCart <: DFTStructure2D end
abstract type DFTStructure3D <: DFTStructure end

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
    ngrid::Int64
    external_field::ExternalFieldModel
    width::Float64
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
    ngrid::Int64
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
    ngrid::Int64
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
    bounds::Array{Float64}
    ngrid::Tuple{Int64}
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
    bounds::Array{Float64}
    ngrid::Tuple{Int64,Int64}
end

export TwoPhase1DCart, Uniform1DCart, ExternalField1DCart
export Uniform1DSphr
export TwoPhase2DLamCart