# Shared by every script in this directory. Run each figure script from the repo root with
# its own environment active, e.g.:
#     julia --project=. examples/docs_figures/01_getting_started.jl
# (add CairoMakie, and whatever else each script's `using` line needs, to that environment
# first: `julia --project=. -e 'using Pkg; Pkg.add("CairoMakie")'`).

const ASSETS = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets"))
mkpath(ASSETS)
assetpath(name) = joinpath(ASSETS, name)
