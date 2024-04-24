push!(LOAD_PATH,"../src/")
using Documenter,cDFT

makedocs(sitename = "cDFT.jl",
format = Documenter.HTML(
    canonical = "https://ClapeyronThermo.github.io/cDFT.jl/"),
warnonly = Documenter.except(),
    authors = "Pierre J. Walker and Andrés Riedemann.",
    pages = [
        "Home" => "index.md",
        "API" => Any[
        "System" => "api/system.md",
        "Methods" => "api/methods.md",
        "Structure" => "api/structure.md",
        "Fields" => "api/fields.md",
        "Profiles" => "api/profiles.md",
        "Utils" => "api/utils.md",
        "Options" => "api/options.md"
        ]])

        deploydocs(;
    repo="github.com/ClapeyronThermo/cDFT.jl.git",
)
