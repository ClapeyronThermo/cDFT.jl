abstract type DFTStructure end 
abstract type DFTStructure1D <: DFTStructure end
abstract type DFTStructure1DCart <: DFTStructure1D end
abstract type DFTStructure1DSphr <: DFTStructure1D end

abstract type DFTStructure2D <: DFTStructure end
abstract type DFTStructure3D <: DFTStructure end

"""
    SurfaceTension1DCart(conditions::Tuple{Float64,Float64,Vector{Float64}}, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate liquid-vapour interfaces in 1D-cartesian coordinates. Contains:
- `conditions`: The x, p, T conditions in the liquid phase at which the calculations are performed.
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

julia> L = length_scale(model)

julia> structure = SurfaceTension1DCart((p, T, [1.]), [-10L, 10L], 201)
```
"""
struct SurfaceTension1DCart <: DFTStructure1DCart 
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
end

"""
    InterfacialTension1DCart(conditions::Tuple{Float64,Float64,Vector{Float64}}, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate liquid-liquid interfaces in 1D-cartesian coordinates. Contains:
- `conditions`: The n, p, T conditions of the total system before it splits between the two liquid phases.
- `bounds`: Specifies the location of the bounds of the system. The interface will be located in the middle.
- `ngrids`: The number of grid points used to represent the density profile.
In the case of the interfacial tension calculation, the composition specified is the total system composition before it splits into two phases. These compositions can be determined ahead of time using Clapeyron:
```julia
julia> model = PCSAFT(["water","hexane"])

julia> (n,p,T) = ([0.5,0.5], 1e5, 298.15)

julia> (x,_,_) = tp_flash(model, p, T, n)
```
As the density profiles in the two phases are expected to reach their bulk values, the densities at the bounds are _fixed_ at their bulk values. In general, it is recommends to use a width of about `20L` (`L` being the length scale of the model) and about 201 grid points.

The profiles will be initialised as generic sigmoidals of the form:
`tanh_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*tanh((x-shift)*coef)+1/2*(start+stop)`

Example:
```julia
julia> L = length_scale(model)

julia> structure = InterfacialTension1DCart((p, T, n), [-10L, 10L], 201)
```
"""
struct InterfacialTension1DCart <: DFTStructure1DCart 
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
    composition_II::Vector{Float64}
end

"""
    Uniform1DCart(conditions::Tuple{Float64,Float64,Vector{Float64}}, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate a uniform system in 1D-cartesian coordinates. Contains:
- `conditions`: The n, p, T conditions of the system.
- `bounds`: Specifies the location of the bounds of the system.
- `ngrids`: The number of grid points used to represent the density profile.
This structure should generally be used to benchmark the DFT code against the bulk calculations.
Example:
```julia
julia> structure = Uniform1DCart((p, T, n), [-10L, 10L], 201)
```
"""
struct Uniform1DCart <: DFTStructure1DCart
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
end

"""
    Uniform1DSphr(conditions::Tuple{Float64,Float64,Vector{Float64}}, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate a uniform system in 1D-spherical coordinates. Contains:
- `conditions`: The n, p, T conditions of the system.
- `bounds`: Specifies the location of the bounds of the system.
- `ngrids`: The number of grid points used to represent the density profile.
This structure should generally be used to benchmark the DFT code against the bulk calculations. As this is spherical coordinates, the lower bound must be greater than 0.
Example:
```julia
julia> structure = Uniform1DSphr((p, T, n), [0L, 10L], 201)
```
"""
struct Uniform1DSphr <: DFTStructure1DSphr
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
end

"""
    InterfacialTension1DSphr(conditions::Tuple{Float64,Float64,Vector{Float64}}, bounds::Vector{Float64}, ngrid::Int64)

The generic structure type used when trying to simulate interfaces in 1D-spherical coordinates. Contains:
- `conditions`: The x, p, T conditions in the bulk phase at which the calculations are performed.
- `bounds`: Specifies the location of the bounds of the system. The interface will be located in the middle.
- `ngrids`: The number of grid points used to represent the density profile.
- `core_composition`: The composition at the core of the spherical coordinates. The density of the system will be determined at the same pressure and temperature of the bulk.
- `r_interface`: Location of the interface (must be between the `bounds`)
In the case of the surface tension calculation, the pressure specified must be the saturation pressure at T and x. As the bound in the bulk should reach the value of the bulk density, it will be treated as fixed. The bound in the core, as its density isn't known ahead of time, will be treated as free. When using a weighted-density approach, it is recommend to not set the lower-bound of the profiles at 0, as these integrals will no longer be well-defined.

The profiles will be initialised as generic sigmoidals of the form:
`tanh_prof(x,start,stop,shift,coef) = 1/2*(start-stop)*tanh((x-shift)*coef)+1/2*(start+stop)`

Example:
```julia
julia> model = PCSAFT(["water","1-octanol"])

julia> (x, p, T) = ([0.9999, 1e-4], 1e5, 298.15)

julia> y = [1e-3, 0.999]

julia> L = length_scale(model)

julia> structure = InterfacialTension1DSphr((p, T, x), [2L, 22L], 201, y, 12L)
```
"""
struct InterfacialTension1DSphr <: DFTStructure1DSphr
    conditions::Tuple{Float64,Float64,Vector{Float64}}
    bounds::Vector{Float64}
    ngrid::Int64
    core_composition::Vector{Float64}
    r_interface::Float64
end

export SurfaceTension1DCart, InterfacialTension1DCart, Uniform1DCart
export Uniform1DSphr, InterfacialTension1DSphr