module LaTeXStringscDFTExt

using cDFT
using LaTeXStrings

const _UNICODE_TO_LATEX = (
    "ρ" => "\\rho",
    "σ" => "\\sigma",
    "ω" => "\\omega",
    "Å" => "\\mathring{A}",
    "³" => "^3",
    "²" => "^2",
)

function cDFT.texlabel(s::String)
    out = s
    for (u, l) in _UNICODE_TO_LATEX
        out = replace(out, u => l)
    end
    return LaTeXStrings.latexstring(out)
end

end
