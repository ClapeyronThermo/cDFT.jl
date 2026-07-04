push!(LOAD_PATH,"../src/")
using Documenter,cDFT

makedocs(sitename = "cDFT.jl",
format = Documenter.HTML(
    canonical = "https://ClapeyronThermo.github.io/cDFT.jl/"),
warnonly = Documenter.except(),
    authors = "Pierre J. Walker and Andrés Riedemann.",
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Tutorials" => Any[
        "Getting Started" => "tutorials/getting_started.md",
        "Choosing a Geometry & Adsorption" => "tutorials/geometries.md",
        "Vapour-Liquid Interfaces" => "tutorials/vapor_liquid_interfaces.md",
        "Multi-Dimensional Interfaces & Copolymer Phases" => "tutorials/multidimensional_interfaces.md",
        "Group-Contribution & Heterosegmented Chains" => "tutorials/group_contribution_chains.md",
        "Copolymer Microphase Morphologies" => "tutorials/copolymer_morphology.md",
        "Electrolytes" => "tutorials/electrolytes.md",
        "Dynamic DFT" => "tutorials/dynamic_dft.md",
        "GPU Acceleration" => "tutorials/gpu_acceleration.md",
        ],
        "Available Models" => Any[
        "SAFT-based Models" => "models/saft.md",
        "Other Functionals" => "models/other.md",
        "Electrolytes" => "models/electrolytes.md",
        ],
        "Structures & External Fields" => Any[
        "Structures" => "structures.md",
        "External Fields" => "external_fields.md",
        ],
        "API" => Any[
        "System" => "api/system.md",
        "Methods" => "api/methods.md",
        "Fields" => "api/fields.md",
        "Free Energy Evaluation" => "api/free_energy.md",
        "Utils" => "api/utils.md",
        "Options" => "api/options.md"
        ],
        "FAQ" => "faq.md",
        ])

        deploydocs(;
    repo="github.com/ClapeyronThermo/cDFT.jl.git",
)
