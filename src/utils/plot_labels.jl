"""
    texlabel(s::AbstractString)

Convert a plot label string (as used by `PlotscDFTExt`/`MakiecDFTExt`) into a LaTeX-rendered
`LaTeXString`. Only implemented once `LaTeXStrings.jl` is loaded — see
`ext/LaTeXStringscDFTExt.jl`. Called when `latex=true` is passed to `plot(system, ...)`.
"""
texlabel(s::AbstractString) = error("latex=true requires LaTeXStrings.jl to be loaded: run `using LaTeXStrings` before plotting.")
export texlabel
