# Multi-Dimensional Interfaces & Copolymer Phases

[Vapour-Liquid Interfaces](@ref) covered the simplest case: a planar interface with a
single coordinate. Real interfaces are often curved (a droplet or bubble) or, for block
copolymers, spontaneously break translational symmetry into periodic microdomains. cDFT's
2D/3D two-phase structures (see [Choosing a Geometry & Adsorption](@ref)) cover both.

## A spherical droplet

[`TwoPhase3DSphrCart`](@ref cDFT.TwoPhase3DSphrCart) embeds a spherically-symmetric
interface in a 3D Cartesian box (GPU-compatible, unlike `Uniform1DSphr`) — the profile is
initialised as a sphere of the first phase surrounded by the second:

```julia
julia> using Clapeyron, cDFT

julia> model = PCSAFT(["water"])

julia> T = 298.15

julia> (p, vl, vv) = saturation_pressure(model, T)

julia> ρl, ρv = [1.0]./vl, [1.0]./vv

julia> L = cDFT.length_scale(model)

julia> ngrid = 51

julia> structure = cDFT.TwoPhase3DSphrCart((p, T), ρl, ρv, [-10L 10L; -10L 10L; -10L 10L], (ngrid, ngrid, ngrid))

julia> system = DFTSystem(model, structure)

julia> ρ = initialize_profiles(system)

julia> converge!(system, ρ)
```

```julia
julia> using CairoMakie

julia> fig = plot(system, ρ)

julia> save("droplet_slice.png", fig)
```

![Cross-section through a converged 3D water droplet density field](../assets/multidimensional_interfaces_droplet.png)

Note that `TwoPhase3DSphrCart`/`TwoPhase2DHexCart`/`TwoPhase3DHexCart` are not currently
exported (construct them with the `cDFT.` prefix as above), unlike the other structure
types.

## Lamellar and hexagonal copolymer phases

[`TwoPhase2DLamCart`](@ref cDFT.TwoPhase2DLamCart)/[`TwoPhase3DLamCart`](@ref cDFT.TwoPhase3DLamCart)
and [`TwoPhase2DHexCart`](@ref cDFT.TwoPhase2DHexCart)/[`TwoPhase3DHexCart`](@ref cDFT.TwoPhase3DHexCart)
follow the same pattern, but are typically used with a block-copolymer functional rather
than a simple VLE pair — a diblock melt naturally microphase-separates into lamellar or
hexagonal domains rather than a single planar interface. Building the copolymer `model`
itself requires the group-contribution machinery covered in
[Group-Contribution & Heterosegmented Chains](@ref) (a `HeterogcPCPSAFT`/`gcPCPSAFT` model
plus a `custom_structure` describing the bead sequence, e.g. `custom_structure("AAAABBBB")`
for a symmetric diblock):

```julia
julia> structure = TwoPhase2DLamCart((p, T), ρ_A_rich, ρ_B_rich, [-10L 10L; -10L 10L], (101, 101))

julia> system = DFTSystem(copolymer_model, structure; mol_structure = Dict("mol" => custom_structure("AAAABBBB")))
```

These `TwoPhase*` structures assume the interface/domain shape up front (a single
planar/cylindrical domain within the box). For seeding and converging genuine
crystallographic microphase morphologies (BCC spheres, a hexagonally-packed cylinder
lattice, gyroid, or a multi-layer lamellar stack) with explicit control over which groups
form which domain, see [Copolymer Microphase Morphologies](@ref).
