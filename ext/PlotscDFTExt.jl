module PlotscDFTExt

using cDFT
using Plots

function Plots.plot(system::cDFT.DFTSystem)
    profiles = system.profiles
    structure = system.structure
    model = system.model
    nc = length(model)

    bounds = structure.bounds

    z = LinRange(bounds[1],bounds[2],structure.ngrid*10)
    L = cDFT.length_scale(model)

    plt = Plots.plot(grid=:off,
                    framestyle=:box,
                    foreground_color_legend = nothing,
                    xtickfontsize=12,
                    ytickfontsize=12,
                    xlabelfontsize=14,
                    ylabelfontsize=14,
                    legend_font=font(12))

    ymax = 0.
    for i in 1:nc
        species = model.components[i]
        Mw = model.params.Mw[i]
        Plots.plot!(plt,z./L,profiles[i].(z)*Mw/1e3,label="$species",linewidth=3)
        ymax = max(ymax,maximum(profiles[i].density)*Mw/1e3)
    end
    Plots.xlims!(plt,bounds./L)
    Plots.ylims!(plt,(0,1.1*ymax))
    Plots.xlabel!(plt,"z / σ")
    Plots.ylabel!(plt,"ρ / (kg/m³)")
    return plt
end

end