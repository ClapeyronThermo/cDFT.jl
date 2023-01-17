![ClapeyronDFT_logo](docs/ClapeyronDFT_logo.svg)
Welcome to ClapeyronDFT! This package intends to provide a comprehensive library of classical Density Functional Theory models, as well as a simple framework to develop your own! ClapeyronDFT uses much of the same tools as Clapeyron and, as such, should be used in conjunction. 

## Example usage
Currently, ClapeyronDFT can be used to obtain surface and interfacial tensions for both pure and mixture systems using the PC-SAFT functionals:

```julia
julia> using Clapeyron, ClapeyronDFT

julia> model = PCSAFT(["water","octane"])
PCSAFT{BasicIdeal} with 2 components:
 "water"
 "octane"
Contains parameters: Mw, segment, sigma, epsilon, epsilon_assoc, bondvol

julia> interfacial_tension(model,1e5,298.15,[0.5,0.5])
0.05104399059834009
```

## Package in active Development
Note that at its current stage, ClapeyronDFT is still in the early stages of development, and things may be moving around or changing rapidly, but we are very excited to see where this project may go!